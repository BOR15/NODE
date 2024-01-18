import torch
import torch.nn as nn
import tensorflow as tf
from time import time

#random stuff file 

# checks if cuda is available and returns chosen and availible device
def check_cuda(use_cuda=False):
    print(f"gpu availible: {torch.cuda.is_available()}")
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using {device}")
    return device

def check_cuda_tensorflow(use_cuda=False):
    """ Don't use it will probably not work if mutiple gpus/cpus are available"""
    
    print(f"gpu availible: {tf.test.is_gpu_available()}")
    if use_cuda:
        device = tf.config.list_physical_devices('GPU')
    else:
        device = tf.config.list_physical_devices('CPU')
    print(f"using {device}")
    return device

# basic timer decorator (to be improved)
def tictoc(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time() - t1
        print(f"Time elapsed for '{func.__name__}': {t2} seconds")
        return result
    return wrapper