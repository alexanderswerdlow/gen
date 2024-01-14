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

logger: Optional[MultiProcessAdapter] = None

def set_logger(name: str):
    global logger
    logger = get_accelerate_logger(name)

def get_logger():
    return logger

def set_log_file(log_file_path):
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method

def get_logger_() -> logging.Logger:
    global logger
    if logger is not None and PartialState._shared_state != {}:
        return logger
    
    from gen.utils.decoupled_utils import is_main_process
    if is_main_process():
        if logger is None:
            return logging.getLogger(__name__)
        elif PartialState._shared_state == {}:
            return logger.logger
    else:
        return Dummy()
        
def log_info(msg, **kwargs):
    get_logger_().info(msg, **kwargs)

def log_error(msg, **kwargs):
    get_logger_().error(msg, **kwargs)

def log_warn(msg, **kwargs):
    get_logger_().warning(msg, **kwargs)