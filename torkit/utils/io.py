import os
import pickle
import gzip
import yaml
import json


def dump_pickle(obj, path):
    path = str(path)
    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))


def load_pickle(path):
    path = str(path)
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))


def load_yaml(path, loader=yaml.SafeLoader):
    with open(path, 'r') as f:
        return yaml.load(f, loader)


def load_json(path, **kwargs):
    with open(path, 'r') as f:
        return json.load(f, **kwargs)


def dump_json(obj, path, indent=1, **kwargs):
    with open(path, 'w') as f:
        return json.dump(obj, f, indent=indent, **kwargs)
