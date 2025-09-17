# infrastructure/log_handlers/console_handler.py

from logging import StreamHandler, Formatter

def get_console_handler(level: str = 'DEBUG'):
    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    return handler
