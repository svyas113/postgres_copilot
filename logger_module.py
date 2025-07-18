import logging
import sys
from pythonjsonlogger import jsonlogger
import memory_module

# --- Configuration ---
LOG_FILE_NAME = 'error_log.json'

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = self.formatTime(record, self.datefmt)
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        
        # Add module, function, and line number details
        log_record['module'] = record.module
        log_record['funcName'] = record.funcName
        log_record['lineno'] = record.lineno

        # If there's exception info, add it as a traceback
        if record.exc_info:
            log_record['traceback'] = self.formatException(record.exc_info)

def get_logger(name='postgres_copilot'):
    """
    Retrieves a logger instance configured to write structured JSON logs.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG) # Set the lowest level to capture all logs

    # --- File Handler ---
    log_directory = memory_module.get_error_log_dir()
    log_file_path = f"{log_directory}/{LOG_FILE_NAME}"
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG) # Log everything to the file
    
    # Use our custom JSON formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Console Handler (optional, for clean console output) ---
    # This handler will only show simple, human-readable messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Only show INFO and above on console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
