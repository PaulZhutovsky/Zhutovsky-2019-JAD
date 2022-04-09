import logging


def get_logger(name='ml_project'):
    return logging.getLogger(name)


def create_logger(log_path='log.log'):
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # file logger will have a higher logging level (i.e. will also log debug messages)
    logger_to_file = logging.FileHandler(log_path)
    logger_to_file.setLevel(logging.DEBUG)

    # console handler will only log info messages
    logger_to_console = logging.StreamHandler()
    logger_to_console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger_to_file.setFormatter(formatter)
    logger_to_console.setFormatter(formatter)

    if not len(logger.handlers):
        logger.addHandler(logger_to_file)
        logger.addHandler(logger_to_console)

    return logger
