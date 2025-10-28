import os
from torch.distributed import init_process_group

def ddp_setup(rank: int, world_size: int):
    # 如果采用 torchrun 运行代码，则无需设定 master 地址和端口
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # rank 标识逻辑 gpu 的序号
    init_process_group(backend="nccl", rank=rank, world_size=world_size)