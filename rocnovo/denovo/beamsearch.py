from abc import ABC, abstractmethod
import torch
import einops
from typing import Union

# import collections

from .denovo import BaseModel, BiDirectRopeClipModel
from ..tokenizer.peptide import SPECIAL_TOKENS, CANONICAL, SOS, PAD, PTMPeptideTokenizer, H2O, PROTON


class BasePredictor(ABC):

    @abstractmethod
    def decode(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        src_padding_mask: torch.BoolTensor,
    ):
        pass


class BeamStat:

    def __init__(
        self,
        tokenizer: PTMPeptideTokenizer,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        batch_size: int,
        num_beams: int = 5,
        max_length: int = 100,
        mass_tolerance: float = 50,
        max_isotope: int = 1,
        min_peptide_len: int = 6,
        is_reverse: bool = False,
        # length_penalty: float = 1,
    ):
        """
        Initialize n-best list of hypotheses.
        """
        # FIXME: 可以将 epoch 和 batch 的初始化分开（耗时减小应该较少）
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_beams = num_beams
        self.mass_tolerance = mass_tolerance
        self.max_isotope = max_isotope
        self.min_peptide_len = min_peptide_len
        self.vocab_size = self.tokenizer.vocab_size

        self.is_reverse = self.tokenizer.reverse if not is_reverse else (not self.tokenizer.reverse)
        # self.length_penalty = length_penalty
        self.batch_size = batch_size
        self.device = tokens.device

        self.precursors = einops.repeat(precursors, "B L -> (B S) L", S=self.num_beams)  # b*s 3 用来进行质量约束
        # self.memory = einops.repeat(memory, "B L D -> (B S) L D", S=self.num_beams)  # b*s l d
        # self.memory_padding_mask = einops.repeat(memory_padding_mask, "B L -> (B S) L", S=self.num_beams)  # b*s l

        # self.cls_token = None
        # if cls_token is not None:
        #     self.cls_token = einops.repeat(cls_token, "B L -> (B S) L", S=self.num_beams)

        self.tokens = torch.full(
            (self.batch_size, self.num_beams, self.max_length),
            fill_value=self.tokenizer.vocab2idx[PAD],
            dtype=torch.int,
            device=self.device,
        )  # b s t_max
        self.tokens[:, :, :1] = einops.repeat(tokens, "B L -> B S L", S=self.num_beams)  # sos 填入
        self.tgt_padding_mask = torch.zeros(
            (self.batch_size * self.num_beams, self.max_length),
            dtype=torch.bool,
            device=self.device,
        )  # b*s t_max 不需要 padding 每次的输入必然是无 pad 的 tokens 部分

        # 每一步生成的有效 beam 带来的所有 vocab 的分数，那一步计算时非活跃的 beam 值为 -inf
        self.beams_scores = torch.full(
            (
                self.batch_size,
                self.num_beams,
                self.max_length,
                self.tokenizer.vocab_size,
            ),
            fill_value=float("-inf"),
            device=self.device,
            dtype=torch.float32,
        )  # b s l v
        self.beams_scores[:, :, 0, :] = 0.0
        # 当前 B S 对应的分数
        self.beams_cur_sum_scores = torch.zeros(
            (self.batch_size, self.num_beams), dtype=torch.float32, device=self.device
        )
        # 表示当前 step 的状态信息，在 update 时更新
        self.active_beams = torch.ones(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)
        self.finished_beams = torch.zeros(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)
        self.discarded_beams = torch.zeros(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)
        self.beam_fits_precursor = torch.zeros(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)

        # 存储已经生成的肽段
        # self.pred_cache = collections.OrderedDict((i, []) for i in range(self.batch_size))
        self.cur_step = 1

        # 提前建立映射
        self.idx2masses = torch.full((self.vocab_size,), fill_value=0.0, dtype=torch.float32, device=self.device)
        for aa, mass in self.tokenizer.masses.items():
            self.idx2masses[self.tokenizer.vocab2idx[aa]] = mass

        self.is_aa_neg_token = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for aa, mass in tokenizer.masses.items():
            if mass < 0:
                self.is_aa_neg_token[tokenizer.vocab2idx[aa]] = True

        self.is_n_term_token = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for aa in tokenizer.masses:
            if aa.startswith(("+", "-")):
                self.is_n_term_token[tokenizer.vocab2idx[aa]] = True

        self.is_special_token = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for token in [SOS, PAD]:
            self.is_special_token[tokenizer.vocab2idx[token]] = True

        self.stop_token = self.tokenizer.vocab2idx[SOS]
        self.pad_token = self.tokenizer.vocab2idx[PAD]
        self.isotope_offsets = torch.tensor([i * 1.00335 for i in range(max_isotope + 1)], device=self.device)

        # finished beams 的 scores、tokens 缓存，每一步都会存储
        self.cached_beams_perstep_mean_scores = torch.full(
            (self.batch_size, self.num_beams, self.max_length),
            fill_value=-2.0,
            device=self.device,
            dtype=torch.float32,
        )
        self.cached_beams_perstep_tokens = torch.full(
            (self.batch_size, self.num_beams, self.max_length, self.max_length),
            fill_value=self.tokenizer.vocab2idx[PAD],
            dtype=torch.int,
            device=self.device,
        )

    def init_state(self, init_score):
        # init_score b v
        # 1. 更新全局分数
        self.beams_scores[:, :, 1, :] = einops.repeat(init_score, "B V -> B S V", S=self.num_beams)
        # 2. 选择 topk beams
        topk_scores, topk_idx = torch.topk(init_score, k=self.num_beams)  # b s
        # 3. 更新当前 tokens
        v_idx = topk_idx  # B S 的 v_idx
        self.tokens[:, :, self.cur_step] = v_idx
        # 更新当前 tokens 列表的平均分数
        self.beams_cur_sum_scores = topk_scores  # B S
        # 4. 检查完成与否
        self._finish_beams()
        self._cache_finished_beams()
        self.finished_beams |= self.discarded_beams
        self.active_beams = ~self.finished_beams
        # finish
        self.cur_step += 1

    def update(self, new_score):
        # new_score: (B*S,V)
        # 1. 更新分数
        active_beams = einops.rearrange(self.active_beams, "(B S) -> B S", S=self.num_beams)
        active_idx = active_beams.nonzero(as_tuple=True)  # Tuple of (B_idx, S_idx)
        self.beams_scores[active_idx[0], active_idx[1], self.cur_step, :] = new_score
        # 2. 获取 topk beams
        self._get_topk_beams()
        # 3. 检查 finished beams,并设置对应 beam 的状态
        self._finish_beams()
        self._cache_finished_beams()
        self.finished_beams |= self.discarded_beams
        self.active_beams = ~self.finished_beams
        # finish
        self.cur_step += 1

    def is_done(self):
        return not self.active_beams.any()

    def package_decoder_input(self):
        active = self.active_beams
        tokens = einops.rearrange(self.tokens[:, :, : self.cur_step], "B S L -> (B S) L")
        tgt_padding_mask = self.tgt_padding_mask[:, : self.cur_step]

        return (
            active,
            tokens,
            tgt_padding_mask,
        )

    def get_top_peptide(self):
        all_batch_scores = einops.rearrange(self.cached_beams_perstep_mean_scores, "B S T -> B (S T)")
        all_batch_tokens = einops.rearrange(self.cached_beams_perstep_tokens, "B S T L -> B (S T) L")
        best_tokens_scores, best_tokens_idx = torch.topk(all_batch_scores, k=1, dim=1)
        batch_idx = torch.arange(self.batch_size, device=self.device)
        best_tokens = all_batch_tokens[batch_idx, best_tokens_idx.squeeze(1), :]
        return best_tokens, best_tokens_scores.squeeze(1)

    def _finish_beams(self):
        # 这个函数中 finished beams 是作为候选结果的，discard beams 是直接舍弃的
        tokens = einops.rearrange(self.tokens, "B S L -> (B S) L")
        current_tokens = tokens[:, self.cur_step]

        self.finished_beams.zero_()
        self.discarded_beams.zero_()
        self.beam_fits_precursor.zero_()

        self.finished_beams = current_tokens == self.stop_token
        self.discarded_beams = current_tokens == self.pad_token

        # FIXME: 检查这里的逻辑问题，与 Casanovo 对比逻辑
        # Check N-terminal modifications
        if self.cur_step > 2:

            if self.is_reverse:
                # 这里reverse 时由于不确定最后一位是 sos 还是 aa 所以需要对 final 和 final-1 检测，之后对 final -1 前检测
                final_pos = torch.full((tokens.shape[0],), self.cur_step, device=self.device)
                final_pos[self.finished_beams] = self.cur_step - 1
                idx = torch.arange(tokens.shape[0], device=self.device)

                is_n_term_final = self.is_n_term_token[tokens[idx, final_pos]]
                is_n_term_prev = self.is_n_term_token[tokens[idx, final_pos - 1]]
                multiple_mods = is_n_term_final & is_n_term_prev

                mask = torch.arange(tokens.shape[1], device=self.device) < final_pos[:, None]
                internal_mods = self.is_n_term_token[tokens] & mask
                internal_mods = internal_mods.any(dim=1)

                self.discarded_beams |= multiple_mods | internal_mods
            else:
                # idx = torch.arange(tokens.shape[0], device=self.device)
                # 检查第一个位置是否是 N-terminal 修饰（这是合法的）
                # first_mod = torch.isin(tokens[idx, first_pos], self.n_term_tokens)
                # 构造 mask，屏蔽掉第一位，检查其它位置是否有 N-terminal 修饰（这是非法的）
                mask = torch.arange(tokens.shape[1], device=self.device)[None, :] > 1
                internal_mods = self.is_n_term_token[tokens] & mask
                internal_mods = internal_mods.any(dim=1)
                self.discarded_beams |= internal_mods

        # Check peptide masses
        # 只生成了一个有效 token 或者 sos/pad 前面检查过，后面的检查也不需要，同时保证运行通畅，不然 peptide_tokens 会生成 []
        if self.cur_step < 2:
            return
        pred_tokens = tokens[:, 1 : self.cur_step + 1]
        has_stop = self.finished_beams
        peptide_tokens = pred_tokens  # 可能包含 sos ，但是计算质量时为 0 所以无影响
        peptide_len = self.cur_step - has_stop.to(torch.long)  # 剔除了 sos

        # valid_mask = ~self.is_special_token[peptide_tokens]
        # peptide_tokens = torch.where(valid_mask,peptide_tokens,0)
        peptide_cur_masses = self.idx2masses[peptide_tokens].sum(1) + H2O  # 高效的向量化索引
        precursor_charges = self.precursors[:, 1].to(torch.long)
        calc_mz = torch.where(
            precursor_charges > 0, (peptide_cur_masses / precursor_charges) + PROTON, peptide_cur_masses
        )
        precursor_mz = self.precursors[:, 2]

        # 不考虑 negative mass 的情况，检查是否 fits mass
        mass_errors = torch.abs(
            (calc_mz[:, None] - (precursor_mz[:, None] - self.isotope_offsets / precursor_charges[:, None]))
            / precursor_mz[:, None]
            * 1e6
        )
        fits_mass = (mass_errors < self.mass_tolerance).any(dim=1)
        # 接下来检查 exceeds mass，需要加上负质量的情况
        exceeds_mass = torch.zeros_like(self.finished_beams, dtype=torch.bool)
        if not self.finished_beams.all():
            neg_token_mask = self.is_aa_neg_token
            neg_tokens = torch.arange(self.vocab_size, device=self.device)[neg_token_mask]
            for neg_token in neg_tokens:
                neg_masses = peptide_cur_masses + self.idx2masses[neg_token]
                neg_mz = torch.where(precursor_charges > 0, (neg_masses / precursor_charges) + PROTON, neg_masses)
                neg_errors = (
                    (neg_mz[:, None] - (precursor_mz[:, None] - self.isotope_offsets / precursor_charges[:, None]))
                    / precursor_mz[:, None]
                    * 1e6
                )
                exceeds_mass |= ~self.finished_beams & (neg_errors > self.mass_tolerance).all(dim=1)

        self.beam_fits_precursor = self.finished_beams & fits_mass & (peptide_len >= self.min_peptide_len)
        self.discarded_beams |= self.finished_beams & (peptide_len < self.min_peptide_len)
        self.finished_beams |= exceeds_mass
        # FIXME: 是否需要这一步，exceeds_mass 表示加上 neg 后没有可能在 tolerence 范围内,如果去除 neg 那更不可能，按理说与 fits_mass 没有重合
        self.beam_fits_precursor |= exceeds_mass & fits_mass

    def _cache_finished_beams(self):
        tokens = einops.rearrange(self.tokens, "B S L -> (B S) L")
        to_cache = self.finished_beams & ~self.discarded_beams
        if not to_cache.any():
            return

        # 获取要缓存的束索引
        cache_indices = to_cache.nonzero(as_tuple=True)[0]
        spec_indices = cache_indices // self.num_beams
        beam_indices = cache_indices % self.num_beams
        # 计算分数
        mean_scores = self.beams_cur_sum_scores[spec_indices, beam_indices] / self.cur_step
        exp_scores = torch.exp(mean_scores)
        peptide_scores = torch.where(self.beam_fits_precursor[cache_indices], exp_scores, exp_scores - 1)
        # 获取肽段 tokenss
        peptide_tokens = tokens[cache_indices]
        # FIXME:这样添加一个 sos token，是否会出现问题， cur_step 可能是 sos 也可能是 aa，所以没问题，且标记为完成，不会参与下一步预测，已经提取出来了，不会有新的变化
        if self.cur_step + 1 < self.max_length:
            peptide_tokens[:, self.cur_step + 1] = self.stop_token
        # 记录当前步骤的结果
        self.cached_beams_perstep_mean_scores[spec_indices, beam_indices, self.cur_step] = peptide_scores
        self.cached_beams_perstep_tokens[spec_indices, beam_indices, self.cur_step, :] = peptide_tokens

    def _get_topk_beams(self):
        # 获取当前步的所有可能 beam 的分数
        step_sum_scores = einops.repeat(
            self.beams_cur_sum_scores,
            "B S -> B S V",
            V=self.tokenizer.vocab_size,
        )
        cur_scores = self.beams_scores[
            :, :, self.cur_step, :
        ]  # 每一步的 finished beam 其 score 没赋值,所以为 -inf,这样的 beam 按理说不会被选择
        step_sum_scores = step_sum_scores + cur_scores

        step_sum_scores = einops.rearrange(step_sum_scores, "B S V -> B (S V)")
        topk_scores, topk_idx = torch.topk(
            step_sum_scores, k=self.num_beams
        )  # B S idx 表示 B S 对应的下一 token的 s_idx*V+v_idx ，获得完整 tokens 需要 cat(self.tokens[b_idx,s_idx,:self.cur_step],v_idx)
        s_idx = topk_idx // self.tokenizer.vocab_size  # B S 的 s_idx
        v_idx = topk_idx % self.tokenizer.vocab_size  # B S 的 v_idx

        # 先填入 prev_tokens
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(self.batch_size), "B -> (B S)", S=self.num_beams)
        self.tokens[:, :, : self.cur_step] = einops.rearrange(
            self.tokens[b_idx, s_idx, : self.cur_step],
            "(B S) L -> B S L",
            S=self.num_beams,
        )
        # 再添加当前 token
        self.tokens[:, :, self.cur_step] = v_idx
        # 更新当前 tokens 列表的平均分数
        self.beams_cur_sum_scores = topk_scores  # B S


class BirdirectBeamStat:
    def __init__(
        self,
        tokenizer: PTMPeptideTokenizer,
        precursors: torch.Tensor,
        memory: torch.Tensor,
        tokens: torch.Tensor,
        tokens_reverse: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        cls_token: torch.Tensor = None,
        num_beams: int = 5,
        max_length: int = 100,
        mass_tolerance: float = 50,
        max_isotope: int = 1,
        min_peptide_len: int = 6,
        # length_penalty: float = 1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_beams = num_beams
        self.mass_tolerance = mass_tolerance
        self.max_isotope = max_isotope
        self.min_peptide_len = min_peptide_len
        self.vocab_size = self.tokenizer.vocab_size
        # self.length_penalty = length_penalty
        self.batch_size = memory.shape[0]
        self.device = tokens.device

        self.beam_stat = BeamStat(
            tokenizer=tokenizer,
            tokens=tokens,
            precursors=precursors,
            batch_size=memory.shape[0],
            num_beams=num_beams,
            max_length=max_length,
            mass_tolerance=mass_tolerance,
            max_isotope=max_isotope,
            min_peptide_len=min_peptide_len,
            is_reverse=False,
        )

        self.beam_stat_reverse = BeamStat(
            tokenizer=tokenizer,
            tokens=tokens_reverse,
            precursors=precursors,
            batch_size=memory.shape[0],
            num_beams=num_beams,
            max_length=max_length,
            mass_tolerance=mass_tolerance,
            max_isotope=max_isotope,
            min_peptide_len=min_peptide_len,
            is_reverse=True,
        )

        self.precursors = einops.repeat(precursors, "B L -> (B S) L", S=self.num_beams)  # b*s 3 用来进行质量约束
        self.memory = einops.repeat(memory, "B L D -> (B S) L D", S=self.num_beams)  # b*s l d
        self.memory_padding_mask = einops.repeat(memory_padding_mask, "B L -> (B S) L", S=self.num_beams)  # b*s l

        self.cls_token = None
        if cls_token is not None:
            self.cls_token = einops.repeat(cls_token, "B L -> (B S) L", S=self.num_beams)

        self.active = torch.ones(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)
        self.active_reverse = torch.ones(self.batch_size * self.num_beams, dtype=torch.bool, device=self.device)

    def init_state(self, init_score, init_score_reverse):
        self.beam_stat.init_state(init_score)
        self.beam_stat_reverse.init_state(init_score_reverse)

    def update(self, new_score, new_score_reverse):
        # TODO: 只提取对应 self.active 为 1 的位置的分数作为输入
        active_final = self.active | self.active_reverse
        valid_pos = self.active[active_final]
        valid_pos_reverse = self.active_reverse[active_final]
        self.beam_stat.update(new_score[valid_pos])
        self.beam_stat_reverse.update(new_score_reverse[valid_pos_reverse])

    def is_done(self):
        return (not self.beam_stat.active_beams.any()) and (not self.beam_stat_reverse.active_beams.any())

    def package_decoder_input(self):
        self.active, tokens, tgt_padding_mask = self.beam_stat.package_decoder_input()
        self.active_reverse, tokens_reverse, tgt_padding_mask_reverse = self.beam_stat_reverse.package_decoder_input()

        active_final = self.active | self.active_reverse

        return (
            self.memory[active_final],
            self.precursors[active_final],
            tokens[active_final],
            tokens_reverse[active_final],
            self.memory_padding_mask[active_final],
            tgt_padding_mask[active_final],
            tgt_padding_mask_reverse[active_final],
            self.cls_token[active_final],
        )

    def get_top_peptide(self):
        tokens, sum_scores = self.beam_stat.get_top_peptide()
        tokens_reverse, sum_scores_reverse = self.beam_stat_reverse.get_top_peptide()
        return tokens, sum_scores, tokens_reverse, sum_scores_reverse


class BeamsearchPredictor(BasePredictor):

    def __init__(
        self,
        model: BaseModel,
        tokenizer: PTMPeptideTokenizer,
        ############## beam search ##############
        beam_size: int = 5,
        max_len: int = 100,
        n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS),
        mass_tolerance: int = 50,
        max_isotope: int = 1,
        min_peptide_len: int = 6,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.n_vocab = n_vocab
        self.mass_tolerance = mass_tolerance
        self.max_isotope = max_isotope
        self.min_peptide_len = min_peptide_len

    def decode(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        src_padding_mask: torch.BoolTensor,
    ):
        self.model.eval()

        with torch.no_grad():
            batch_size = spectra.shape[0]
            max_length = self.max_len + 2
            n_beam = self.beam_size

            memory, memory_padding_mask, cls_token, _ = self.model.encode(
                spectra=spectra,
                precursors=precursors,
                src_padding_mask=src_padding_mask,
            )  # (B, L , model_dim)

            tokens = torch.full(
                (batch_size, 1),
                fill_value=self.tokenizer.vocab2idx[SOS],
                dtype=torch.int,
            ).to(memory.device)
            tgt_padding_mask = torch.full((batch_size, 1), fill_value=False, dtype=torch.bool).to(memory.device)

            tokens_reverse = torch.full(
                (batch_size, 1),
                fill_value=self.tokenizer.vocab2idx[SOS],
                dtype=torch.int,
            ).to(memory.device)
            tgt_padding_mask_reverse = torch.full((batch_size, 1), fill_value=False, dtype=torch.bool).to(memory.device)

            # TODO: 初始化 beam stat
            beam_stat = BirdirectBeamStat(
                tokenizer=self.tokenizer,
                precursors=precursors,
                memory=memory,
                tokens=tokens,
                tokens_reverse=tokens_reverse,
                cls_token=cls_token,
                memory_padding_mask=memory_padding_mask,
                num_beams=n_beam,
                max_length=max_length,
                mass_tolerance=self.mass_tolerance,
                max_isotope=self.max_isotope,
                min_peptide_len=self.min_peptide_len,
                # length_penalty=0.1,
            )
            # TODO: 添加初始的 beam

            init_score, init_score_reverse = self.model.candidates_scores(
                memory=memory,
                tokens=tokens,
                tokens_reverse=tokens_reverse,
                precursors=precursors,
                memory_padding_mask=memory_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                tgt_reverse_padding_mask=tgt_padding_mask_reverse,
                cls_token=cls_token,
            )  # (B, V) 由 sos 起始字符生成

            beam_stat.init_state(init_score, init_score_reverse)

            # TODO: 进入循环
            for step in range(2, max_length):
                if beam_stat.is_done():
                    break
                (
                    beam_memory,
                    beam_precursors,
                    beam_tokens,
                    beam_tokens_reverse,
                    beam_memory_padding_mask,
                    beam_tgt_padding_mask,
                    beam_tgt_padding_mask_reverse,
                    beam_cls_token,
                ) = beam_stat.package_decoder_input()
                new_score, new_score_reverse = self.model.candidates_scores(
                    memory=beam_memory,
                    tokens=beam_tokens,
                    tokens_reverse=beam_tokens_reverse,
                    precursors=beam_precursors,
                    memory_padding_mask=beam_memory_padding_mask,
                    tgt_padding_mask=beam_tgt_padding_mask,
                    tgt_reverse_padding_mask=beam_tgt_padding_mask_reverse,
                    cls_token=beam_cls_token,
                )  # (B*S,V)
                beam_stat.update(new_score, new_score_reverse)

            # TODO: 整理并返回输出
            return beam_stat.get_top_peptide()


def generate_look_ahead_mask(seq_len: int) -> torch.Tensor:
    # 生成一个上三角矩阵，True 表示 mask 掉
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    return mask  # shape: (seq_len, seq_len)


class GreedysearchPredictor(BasePredictor):
    def __init__(
        self,
        model: BaseModel,  # BaseModel
        tokenizer: PTMPeptideTokenizer,  # PTMPeptideTokenizer
        beam_size: int = 0,
        max_len: int = 100,
        n_vocab: int = len(CANONICAL) + len(SPECIAL_TOKENS),
        mass_tolerance: int = 50,
        max_isotope: int = 1,
        min_peptide_len: int = 6,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_vocab = n_vocab
        self.mass_tolerance = mass_tolerance
        self.max_isotope = max_isotope
        self.min_peptide_len = min_peptide_len

    def decode(self, spectra, precursors, src_padding_mask):
        self.model.eval()
        with torch.no_grad():
            batch_size = spectra.shape[0]
            max_length = self.max_len + 2  # +2 for SOS and EOS

            # 编码输入
            memory, memory_padding_mask, cls_token, _ = self.model.encode(
                spectra=spectra,
                precursors=precursors,
                src_padding_mask=src_padding_mask,
            )  # memory: (B, L, D)

            # 初始化 target tokens（全为SOS）
            tokens = torch.full(
                (batch_size, 1),
                fill_value=self.tokenizer.vocab2idx[SOS],
                dtype=torch.long,
                device=memory.device,
            )  # (B, 1)
            tokens_reverse = torch.full(
                (batch_size, 1),
                fill_value=self.tokenizer.vocab2idx[SOS],
                dtype=torch.long,
                device=memory.device,
            )  # (B, 1)
            sum_scores = torch.zeros(batch_size, dtype=torch.float32, device=memory.device)  # (B,)
            sum_scores_reverse = torch.zeros(batch_size, dtype=torch.float32, device=memory.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
            finished_reverse = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)

            for step in range(1, max_length):  # 从 step=1 开始（已有一个SOS）
                # tgt_mask = generate_look_ahead_mask(step).to(memory.device)  # (step, step)
                tgt_padding_mask = torch.zeros((batch_size, step), dtype=torch.bool, device=memory.device)
                tgt_reverse_padding_mask = torch.zeros((batch_size, step), dtype=torch.bool, device=memory.device)

                # 得到下一个 token 的概率分布
                scores, scores_reverse = self.model.candidates_scores(
                    memory=memory,
                    tokens=tokens,
                    tokens_reverse=tokens_reverse,
                    precursors=precursors,
                    memory_padding_mask=memory_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    tgt_reverse_padding_mask=tgt_reverse_padding_mask,
                    # tgt_mask=tgt_mask,
                    cls_token=cls_token,
                )  # (B, V)

                next_token = torch.argmax(scores, dim=1).unsqueeze(1)  # (B, 1)
                next_token_reverse = torch.argmax(scores_reverse, dim=1).unsqueeze(1)
                tokens = torch.cat([tokens, next_token], dim=1)  # (B, step+1)
                tokens_reverse = torch.cat([tokens_reverse, next_token_reverse], dim=1)
                sum_scores += scores.gather(1, next_token).squeeze(1)  # (B,)
                sum_scores_reverse += scores_reverse.gather(1, next_token_reverse).squeeze(1)

                finished |= next_token.squeeze(1) == self.tokenizer.vocab2idx[SOS]
                finished_reverse |= next_token_reverse.squeeze(1) == self.tokenizer.vocab2idx[SOS]
                if finished.all() and finished_reverse.all():
                    break

            return tokens, sum_scores, tokens_reverse, sum_scores_reverse  # 返回去掉 SOS 的 tokens 和分数
