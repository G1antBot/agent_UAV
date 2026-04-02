import logging
import os
from datetime import datetime

_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
_INITIALIZED = False


class _RunIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "run_id"):
            record.run_id = _RUN_ID
        return True


def init_runtime_logger(log_dir=None, level=logging.INFO):
    global _INITIALIZED

    if _INITIALIZED:
        return _RUN_ID

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"run_{_RUN_ID}.log")

    root = logging.getLogger("uav_agent")
    root.setLevel(level)
    root.propagate = False

    run_filter = _RunIdFilter()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(run_filter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(run_filter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    _INITIALIZED = True
    root.info(f"日志系统已初始化，日志文件: {log_path}")
    return _RUN_ID


def get_runtime_logger(name):
    return logging.getLogger(f"uav_agent.{name}")


def get_run_id():
    return _RUN_ID
