import tensorflow as tf
import keras
from tfdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt
# from tools.toydata_processing import val_shift_split
# from tools.plots import *

from keras.layers import Dense
from keras.activations import tanh
# odeint(func, yo, t)

# class odeint, build model.
class ODEfunctens(keras.Model):
    
    def __init__(self, N_feat, N_neurons):
        super(ODEfunctens, self).__init__()
        self.net = keras.models.Sequential()
        self.net.add(keras.Input((N_feat,)))
        self.net.add(Dense(N_neurons, activation=tanh, kernel_initializer='glorot_normal', bias_initializer='zeros',)) # glorot_normal = initializes weight to normal distribution.
        # self.net.add(tanh((N_neurons,)))
        self.net.add(Dense(N_neurons, kernel_initializer='glorot_normal', bias_initializer='zeros',))
        self.net.compile()
    
    def forward(self, t, y):
        y = self.net(y)
        return y
