import logging
from typing import Optional
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format= "%(message)s", datefmt="[%X]", handlers=[RichHandler()])

logger: Optional[logging.Logger] = None


def set_logger(name: str):
    from gen.utils.decoupled_utils import get_rank, is_main_process
    global logger

    logger = logging.getLogger(name if is_main_process() else name + f"_rank_{get_rank()}")
    logger.handlers = []

    console_handler = RichHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def get_logger():
    return logger


def set_log_file(log_file_path):
    log_format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method


def get_logger_(main_process_only: bool) -> logging.Logger:
    from gen.utils.decoupled_utils import get_rank, is_main_process

    if is_main_process() or not main_process_only:
        if logger is not None:
            return logger
        else:
            return logging.getLogger(__name__ if not main_process_only else __name__ + f"_rank_{get_rank()}")
    else:
        return Dummy()

def _always_debug_log(msg, **kwargs) -> logging.Logger:
    from gen.utils.decoupled_utils import is_main_process
    if not is_main_process():
        get_logger_(main_process_only=False).debug(msg, **kwargs)

def log_debug(msg, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(msg, **kwargs)
    get_logger_(main_process_only=main_process_only).debug(msg, **kwargs)


def log_info(msg, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(msg, **kwargs)
    get_logger_(main_process_only=main_process_only).info(msg, **kwargs)


def log_error(msg, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(msg, **kwargs)
    get_logger_(main_process_only=main_process_only).error(msg, **kwargs)


def log_warn(msg, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(msg, **kwargs)
    get_logger_(main_process_only=main_process_only).warning(msg, **kwargs)
