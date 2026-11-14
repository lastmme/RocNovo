# Configuration and runtime notes

## YAML files

The code repository contains `clip.yaml` and `denovo.yaml` as starting points. For a released model, use its paired configuration files. Model dimensions, vocabulary, and tokenizer settings must match the checkpoint.

- `trainer_data.train_path` / `valid_path`: annotated HDF5 inputs.
- `predictor_data.test_path`: inference HDF5 input.
- `trainer.model_save_folder`: checkpoint output folder.
- `trainer.summarywriter`: TensorBoard folder.
- `trainer.max_epochs`: total training epochs.
- `trainer.warmup_epochs`: warmup duration in epochs (may be fractional).
- `trainer.grad_norm_clip`: gradient norm clipping threshold.
- `trainer.validation_steps`: validation interval in steps.
- `trainer.save_top_k`: retained best checkpoints.
- `trainer_data.batch_size` / `predictor_data.batch_size`: training/inference batch sizes.

Set all file paths for your environment; YAML relative paths are resolved against the current working directory.

## Loading mode

| trainer.load_mode | Behavior |
| --- | --- |
| TFS | Start training from scratch |
| RESUME | Restore model, optimizer, and scheduler state |
| WARM_START | Load model weights without resuming optimizer/scheduler state |

For Stage Two, first supply a valid pretrained Stage One checkpoint through `clip_checkpoint_path`. The contrastive model is frozen; freezing randomly initialized CLIP is not the intended pretraining workflow.

## Beam search

`predictor.beam_search` controls `beam_size`, `max_len`, `mass_tolerance`, `max_isotope`, and `min_peptide_len`. Mass tolerance is in **ppm**. The released evaluation examples use beam size 10. See [models](models.md) for loading and [datasets](datasets.md) for the separate V2 evaluation filter.

## Distributed training

Training uses PyTorch distributed processing with the NCCL backend. Use a compatible Linux/CUDA environment. The helper in `rocnovo/common/ddp.py` sets `MASTER_ADDR=localhost` and `MASTER_PORT=12355`. If that port is occupied, select an available port in the helper before launching a new job. GPU rank identifies the local process/device.

## Output columns

`seq1`/`seq2` and their scores are directional predictions; their direction follows the tokenizer's reverse setting. `pred_peptide`/`pred_score` identify the selected output. `idx` retains the original input sample position. Keep this field intact when aligning references or applying V2 exclusions.
