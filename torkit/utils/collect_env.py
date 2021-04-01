# Copyright (c) Facebook, Inc. and its affiliates.
import time
import socket
from torch.utils.collect_env import get_pretty_env_info


def get_PIL_version():
    try:
        import PIL
    except ImportError:
        return '\n No Pillow is found.'
    else:
        return '\nPillow ({})'.format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_PIL_version()
    return env_str


def get_run_name():
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    return run_name
