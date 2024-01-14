import logging
from typing import Optional
from accelerate.logging import get_logger as get_accelerate_logger, MultiProcessAdapter
from accelerate.state import PartialState

log_format = format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
date_format = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(
    format=log_format,
    datefmt=date_format,
    level=logging.INFO,
)

logger: Optional[logging.Logger] = None

def set_logger(name: str):
    from gen.utils.decoupled_utils import get_rank, is_main_process
    global logger
    logger = logging.getLogger(name if is_main_process() else name + f"_{get_rank()}")

def get_logger():
    return logger

def set_log_file(log_file_path):
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method

def get_logger_(main_process_only: bool) -> logging.Logger:
    from gen.utils.decoupled_utils import get_rank, is_main_process
    if is_main_process() or main_process_only:
        if logger is not None:
            return logger
        else:
            return logging.getLogger(__name__ if is_main_process() else __name__ + f"_{get_rank()}")
    else:
        return Dummy()
        
def log_info(msg, main_process_only: bool = True, **kwargs):
    get_logger_(main_process_only=main_process_only).info(msg, **kwargs)

def log_error(msg, main_process_only: bool = True, **kwargs):
    get_logger_(main_process_only=main_process_only).error(msg, **kwargs)

def log_warn(msg, main_process_only: bool = True, **kwargs):
    get_logger_(main_process_only=main_process_only).warning(msg, **kwargs)