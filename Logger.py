import os
import logging
import inspect

class Logger:
    # Class variable to hold the filename, shared among all instances
    log_file_name = 'log.log'

    def __init__(self):
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_path = os.path.join(log_dir, Logger.log_file_name)

        # Configure logging to use the shared log file name
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def get_caller_info(self):
        stack = inspect.stack()
        _, filename, lineno, _, _, _ = stack[2]
        filename = os.path.basename(filename)
        return filename, lineno

    def info(self, message):
        filename, lineno = self.get_caller_info()
        logging.info(f'{filename}:{lineno}: {message}')

    def debug(self, message):
        filename, lineno = self.get_caller_info()
        logging.debug(f'{filename}:{lineno}: {message}')

    def warning(self, message):
        filename, lineno = self.get_caller_info()
        logging.warning(f'{filename}:{lineno}: {message}')

    def error(self, message):
        filename, lineno = self.get_caller_info()
        logging.error(f'{filename}:{lineno}: {message}')

# Create a logger instance that can be imported and used in other modules
logger = Logger()
