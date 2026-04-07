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


class _ConsoleFormatter(logging.Formatter):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[31m",
    }

    EVENT_COLORS = {
        "CMD_START": "\033[36m",
        "CMD_END": "\033[32m",
        "CLAUSE_START": "\033[36m",
        "CLAUSE_ROUTE": "\033[37m",
        "CLAUSE_OK": "\033[32m",
        "CLAUSE_FAIL": "\033[31m",
        "开始执行": "\033[36m",
        "执行结束": "\033[32m",
        "步骤开始": "\033[36m",
        "步骤执行方式": "\033[37m",
        "步骤完成": "\033[32m",
        "步骤失败": "\033[31m",
    }

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        base = super().format(record)
        if not self.use_color:
            return base

        level_color = self.COLORS.get(record.levelname, "")
        if level_color:
            base = base.replace(
                f"| {record.levelname} |",
                f"| {self.BOLD}{level_color}{record.levelname}{self.RESET} |",
                1,
            )

        for event, color in self.EVENT_COLORS.items():
            if event in base:
                base = base.replace(event, f"{self.BOLD}{color}{event}{self.RESET}", 1)
                break

        return base


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

    use_color = os.getenv("UAV_LOG_COLOR", "1") != "0"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_ConsoleFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s",
        use_color=use_color,
    ))
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
