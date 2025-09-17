# infrastructure/log_handlers/file_handler.py

from logging import FileHandler, Formatter

def get_file_handler(path: str = "review.log", level: str = 'INFO'):
    handler = FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    return handler
