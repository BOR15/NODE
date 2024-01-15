import tensorflow as tf
import keras
from tfdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tensorflow_datasets as tfds
import numpy as np
from tools.toydata_processing import val_shift_split
from tools.plots import *
from time import perf_counter as time

from keras.layers import Dense
from keras.activations import tanh
# odeint(func, yo, t)

# class odeint, build model.
class ODEfunctens(keras.Model):
    
    def __init__(self, N_feat, N_neurons):
        super(ODEfunctens, self).__init__()
        self.net = keras.models.Sequential()
        self.net.add(keras.Input(shape=(N_feat,)))
        self.net.add(Dense(N_neurons, activation=tanh, kernel_initializer='glorot_normal', bias_initializer='zeros',)) # glorot_normal = initializes weight to normal distribution.
        self.net.add(Dense(N_feat, kernel_initializer='glorot_normal', bias_initializer='zeros',))
        self.net.compile()
    
    @tf.function
    def call(self, t, y):
        y = self.net(y)
        return y


def main(num_neurons=50, num_epochs=50, learning_rate=0.01, train_duration=1.5, val_shift=0.1):
    for_torch = False

    """
    This function trains a Tensorflow.keras model using the given parameters and data.
    
    Args:
        num_neurons (int): Number of neurons in the model (default: 50).
        num_epochs (int): Number of training epochs (default: 50).
        learning_rate (float): Learning rate for the optimizer (default: 0.01).
        train_duration (float): Duration of the training data (default: 1.5).
        val_shift (float): Shift for the validation data (default: 0.1).
    """

    # defining training and validation loss lists
    train_losses = []
    val_losses = []

    # loading in the data, maybe this needs to be done through tensorflow (tfds.load but this creates a tensorflow dataset)
    loaded_data = np.load('Input_Data/tensors.npz')
    data = (tf.convert_to_tensor(loaded_data['t']), tf.convert_to_tensor(loaded_data['features']))
    train_data, val_data, test_data = val_shift_split(data, train_dur=train_duration, val_shift=val_shift, for_torch=for_torch)

    # paramters
    num_feat = data[1][1].shape[0]

    # defining model, loss and optimizer
    model = ODEfunctens(num_feat, num_neurons)
    MSEloss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    start = time()
    print(train_data[1][0].shape, train_data[0].shape)

    for epoch in range(num_epochs):
        
        # initialize ech optimizer to 0 for each epoch
        with tf.GradientTape() as tape:
            # make predictions
            y_pred = odeint(model, tf.reshape(train_data[1][0],[1,2]), train_data[0]) 

            # get the loss value
            loss = MSEloss(y_pred, train_data[1])

        # gradient of trainable weights with resprect to loss
        # done using gradient tape.
        grads = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #calculate validation loss
        # with tf.stop_gradient(model.trainable_weights):
        y_pred_val = odeint(model, tf.reshape(val_data[1][0], [1,2]), val_data[0])
        val_loss  = MSEloss(y_pred_val, val_data[1])
        
        # store epoch values in list
        train_losses.append(loss)
        val_losses.append(val_loss)
    #training time
    print(f"Training time: {time() - start} seconds")

    #prediction after training is complete
    # with tf.stop_gradient(model.trainable_weights):
    predicted = odeint(model, tf.reshape(data[1][0], [1,2]), data[0])
    eval_loss = MSEloss(predicted, data[1])
    print(f'"Mean Squared Error Loss: {eval_loss}')
    print(predicted.shape)
    # print(data,shape)
    # Plotting the losses
    toy = True
    for_torch = False
    plot_data(data, toy=toy)
    plot_actual_vs_predicted_full(data, tf.reshape(predicted, [1200,2]), toy=toy, for_torch=for_torch)
    # plot_phase_space(data, tf.reshape(predicted, [1200,2]))
    plot_training_vs_validation([train_losses, val_losses], share_axis=True)
    plt.show()

if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass