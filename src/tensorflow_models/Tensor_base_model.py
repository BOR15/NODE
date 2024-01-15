import tensorflow as tf
from tfdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt
from tools.toydata_processing import val_shift_split
from tools.plots import *
# odeint(func, yo, t)

# class odeint, build model.
class ODEint(tf.keras.Model):
    
    def __init__(self, N_feat, N_neurons):
        