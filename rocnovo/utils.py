"""Small utility functions"""

import logging
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import random
import numpy as np

logger = logging.getLogger(__name__)


# 默认日志配置
def setup_logging(log_file: str = None, log_level: str = "INFO"):
    # 配置日志级别
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    # 配置警告捕获
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    warnings_logger = logging.getLogger("py.warnings")
    # 创建不同的格式化器
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} : {message}",
        style="{",
    )
    # 清除现有的处理器
    root_logger.handlers = []
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[log_level.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)
    # 如果指定了日志文件，添加文件处理器
    if log_file is not None:
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        warnings_logger.addHandler(file_handler)
    # 禁用依赖库的非关键日志消息
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

def set_seed(seed):
    seed = seed * dist.get_rank()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True