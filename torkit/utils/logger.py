import sys
from loguru import logger


def setup_logger(filename=None, level='INFO'):
    fmt = '<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>{message}</level>'
    logger.remove()
    logger.add(sys.stderr, level=level, format=fmt)
    if filename:
        logger.add(filename, level=level, format=fmt)

    # To avoid conflict with Tensorflow
    # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
    # logger.propagate = False

    return logger
