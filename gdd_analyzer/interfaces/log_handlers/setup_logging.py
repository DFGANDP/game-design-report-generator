# bootstrap.py

import logging
from logging import Logger, Handler

def setup_logger(name: str = "review-analyzer", handlers: list[Handler] = None) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # Unikamy duplikacji handlerów
        handlers = handlers or []

        for handler in handlers:
            logger.addHandler(handler)

    return logger
