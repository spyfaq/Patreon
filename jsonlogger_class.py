import logging, json, os

class JSONLogger:
    def __init__(self, log_file=None, log_level=logging.INFO, log_dir=None):
        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a log formatter for JSON
        json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "info": "%(info)s"}')

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