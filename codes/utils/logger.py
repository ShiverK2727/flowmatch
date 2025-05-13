import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(log_dir,
                 log_name=None,
                 log_level=logging.DEBUG,
                 max_file_size=15 * 1024 * 1024,  # 15MB
                 backup_count=5):
    """
    Sets up logging with file rotation and no console output.
    :param log_dir: Directory where log file will be saved
    :param log_name: Custom name for the log file (optional)
    :param log_level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
    :param max_file_size: Maximum size of a log file in bytes (default: 10MB)
    :param backup_count: Number of backup log files to keep (default: 5)
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # 在日志文件名中添加时间信息
    if log_name is None:
        log_file = os.path.join(log_dir, f'log_{current_time}.txt')
    else:
        log_file = os.path.join(log_dir, f'{log_name}_{current_time}.txt')

    # 创建一个新的日志记录器，并确保它是唯一的
    logger = logging.getLogger("only_file_logger")

    # 移除任何现有的处理器，确保没有重复的处理器
    logger.handlers.clear()

    logger.setLevel(log_level)

    # 创建一个文件处理器，并设置日志格式
    file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # 将文件处理器添加到日志记录器
    logger.addHandler(file_handler)

    return logger


def log_info(message, print_message=False):
    """
    Logs an info message.
    :param message: Info message to log
    :param print_message: Whether to print the message to console
    """
    logger = logging.getLogger("only_file_logger")  # 使用全局唯一的日志记录器
    logger.log(level=logging.INFO, msg=message)
    if print_message:
        print(message)


def log_error(message, exc_info=None):
    """
    Logs an error message, optionally with exception information.
    :param message: Error message to log
    :param exc_info: Exception information to log (optional)
    """
    logger = logging.getLogger("only_file_logger")  # 使用全局唯一的日志记录器
    logger.error(message, exc_info=exc_info)
    print(message)
