import sys
import os
from contextlib import contextmanager
from omegaconf import OmegaConf

@contextmanager
def extend_sys_path(path):
    original_sys_path = sys.path.copy()
    sys.path.append(path)
    try:
        yield
    finally:
        sys.path = original_sys_path

def returnRepoConfig(configName):
    cwd = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(cwd)

    with extend_sys_path(parent_dir):
        from ConfigurationCreator import create_config

    # Once the context block is exited, sys.path is restored
    config = create_config(repositoryConfigPath= os.path.join(cwd, configName))
    return config