#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging, json, os

class _SafeJSONFormatter(logging.Formatter):
    """Builds each log line via json.dumps instead of a hand-rolled format
    string. The previous formatter built JSON via
    '{"message": "%(message)s", ...}' string substitution, which produces
    INVALID JSON the moment a message or `info` value contains a quote,
    backslash, or newline (e.g. any exception message with a quoted value,
    or an HTTP error body) -- a very plausible occurrence for exactly the
    kind of strings this logger is used to record.
    """
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "info": getattr(record, "info", None),
        }
        return json.dumps(payload, default=str)


class JSONLogger:
    def __init__(self, log_file=None, log_level=logging.INFO, log_dir=None):
        # Use a logger name unique to this log file/dir instead of the
        # shared module-level __name__. logging.getLogger(name) returns the
        # SAME logger object for a given name, so with a shared name every
        # new JSONLogger(...) created in the same process kept adding more
        # handlers to the same underlying logger -- causing each log line to
        # be printed/written multiple times over. Guard further by skipping
        # handler setup entirely if this named logger already has handlers.
        logger_name = f"{__name__}.{log_dir or ''}.{log_file or 'default'}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        if self.logger.handlers:
            return

        json_formatter = _SafeJSONFormatter()

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)

        # Determine the log file path and create the directory if necessary
        if log_file:
            if not log_dir:
                log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Ensure the log file has a .json extension
            if not log_file.endswith('.json'):
                log_file += '.json'
            
            file_path = log_dir + '/' + log_file
            # Create a file handler for logging to the file
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

    def log(self, level, message, info=None):
        if level == 'debug':
            self.logger.debug(message, extra={'info': info})
        elif level == 'info':
            self.logger.info(message, extra={'info': info})
        elif level == 'warning':
            self.logger.warning(message, extra={'info': info})
        elif level == 'error':
            self.logger.error(message, extra={'info': info})
        elif level == 'critical':
            self.logger.critical(message, extra={'info': info})
        else:
            raise ValueError("Invalid log level")