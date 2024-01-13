import torch
import torch.nn as nn
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

# basic timer decorator (to be improved)
def tictoc(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time() - t1
        print(f"Time elapsed for '{func.__name__}': {t2} seconds")
        return result
    return wrapper