import os
from torch.distributed import init_process_group

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # rank 标识逻辑 gpu 的序号
    init_process_group(backend="nccl", rank=rank, world_size=world_size)