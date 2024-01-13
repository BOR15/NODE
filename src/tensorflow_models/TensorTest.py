import tensorflow as tf
from tfdiffeq import odeint
import pandas as pd
import numpy as np

"""
a lot of this garbage is outdated, go look at pytorch versions for more up to date code
some of the code in this file is replaced by functions that should work in tensorflow aswell i think
"""

print(f"using TensorFlow version {tf.__version__}")

data = pd.read_csv("toydatafixed.csv", delimiter=';')
t_tensor = tf.convert_to_tensor(data['t'].values, dtype=tf.float32)
features_tensor = tf.convert_to_tensor(data[['speed', 'angle']].values, dtype=tf.float32)
min_vals = tf.reduce_min(features_tensor, axis=0)
max_vals = tf.reduce_max(features_tensor, axis=0)
features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
num_feat = features_tensor.shape[1]

def simple_split(train_dur, val_dur):
    split_train_dur = train_dur
    split_val_dur = val_dur

    split_train = int(split_train_dur / 0.005)  
    split_val = int((split_val_dur + split_train_dur) / 0.005)
    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
    test_data = (t_tensor[split_val:], features_tensor[split_val:])

def val_shift_split(train_dur, val_shift):
    split_train_dur = train_dur
    shift_val_dur = val_shift

    split_train = int(split_train_dur / 0.005)  
    shift_val = int(shift_val_dur  / 0.005)

    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[shift_val:split_train + shift_val], features_tensor[shift_val:split_train + shift_val])
    test_data = (t_tensor[split_train:], features_tensor[split_train:])
    return train_data, val_data, test_data


class ODEFunc(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ODEFunc, self).__init__(**kwargs)

        self.x = tf.keras.layers.Dense(50, activation='tanh',
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.y = tf.keras.layers.Dense(2,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

    def call(self, t, y):
        y = tf.cast(y, tf.float32)
        x = self.x(y ** 3)
        y = self.y(x)
        return y


#generated with copilot and not adjust for NODE use
def train_model(data, data2,lr = 0.01, num_epochs=10):
    
    t, features = data
    val_t, val_features = data2


    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    net = ODEFunc()

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            pred_y = odeint(net, features[0], t)
            loss = loss_fn(pred_y, features)

        grads = tape.gradient(loss, net.variables)
        grad_vars = zip(grads, net.variables)
        optimizer.apply_gradients(grad_vars)


        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")



# train_data, val_data, test_data = simple_split(2, 2)  # Assuming you want to use the simple_split() function
train_data, val_data, test_data = val_shift_split(3, .3)  # Assuming you want to use the val_shift_split() function

train_model(train_data, val_data)

