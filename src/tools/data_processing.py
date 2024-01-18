import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

"""
READ THIS:

This stuff hasnt been worked on yet, right now its just an old copy of toydata_processing.py 

I have now started on this but still mess 
"""

# Testing testing how to commit and push 
berend_path = r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\Raw_Data\Dynamics40h17.csv"
boris_path = "NODE/Input_Data/Raw_data/Dynamics40h17.csv"
#laetitia_path = "/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/Input_Data/Raw_Data/Dynamics40h17.csv"

berend_scufed = r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\real_data_scuffed2.pt"
boris_scufed = "real_data_scuffed2.pt"

def load_data(filename= berend_path , shift=0, start=300):
    # Import data
    data = pd.read_csv(filename, delimiter=',')

    # Check for duplicates
    duplicates = data.duplicated(subset=['t'])
    if duplicates.any():
        print("Duplicates found in the time axis. Removing duplicates...")
        #data = data.drop_duplicates(subset=['t'])

        data = data.groupby('t').mean().reset_index()
    else:
        print("No duplicates found in the time axis.")

    # Defining tensors
    t_tensor = torch.tensor(data.iloc[start:, 1].values, dtype=torch.float32)
    features_tensor = torch.tensor(data.iloc[start:, 2+shift:27+shift:5].values, dtype=torch.float32)
    print(features_tensor.shape, t_tensor.shape)
    print(features_tensor[0], t_tensor[0])
    features_tensor = normalize_data_mean_0(features_tensor)
    return t_tensor, features_tensor

def normalize_data(features_tensor):
    #normalizing features between 0 and 1
    #It has been found that normalising the data between 0 and 1 is the best for neural networks
    min_vals = torch.min(features_tensor, dim=0)[0]
    print(min_vals)
    max_vals = torch.max(features_tensor, dim=0)[0]
    features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
    return features_tensor

def normalize_data_mean_0(features_tensor, for_torch=True):
    #normalizing features between 0 and 1
    if for_torch:
        mean_vals = torch.mean(features_tensor, dim=0)
        std_vals = torch.std(features_tensor, dim=0)
    else:
        mean_vals = tf.reduce_mean(features_tensor, axis=0)
        std_vals = tf.math.reduce_std(features_tensor, axis=0)
    
    features_tensor = (features_tensor - mean_vals) / std_vals
    return features_tensor

def get_timestep(t_tensor):
    timestep = torch.min(t_tensor[1:] - t_tensor[:-1])
    return timestep


def simple_split(data_tuple, train_dur, val_dur=0, timestep=None):
    t_tensor, features_tensor = data_tuple
    
    if not timestep:
        timestep = get_timestep(t_tensor)
    
    split_train = int(train_dur / timestep)
    split_val = int((val_dur + train_dur) / timestep)

    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
    test_data = (t_tensor[split_val:], features_tensor[split_val:])

    print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]}, test size: {test_data[0].shape[0]}")

    return train_data, val_data, test_data


def val_shift_split(data_tuple, train_dur, val_shift, timestep=None):
    t_tensor, features_tensor = data_tuple

    if not timestep:
        timestep = get_timestep(t_tensor)

    #time to index
    split_train = int(train_dur / timestep)  
    shift_val = int(val_shift  / timestep)

    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[shift_val:split_train + shift_val], features_tensor[shift_val:split_train + shift_val])
    test_data = (t_tensor[split_train:], features_tensor[split_train:])

    print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]} starting at {shift_val}, test size: {test_data[0].shape[0]}")

    return train_data, val_data, test_data


# with seperate axis but not working
# def plot_data(data_tuple):
#     time_points = data_tuple[0].numpy()  
#     feature_data = data_tuple[1].numpy() 

#     fig, ax1 = plt.subplots(figsize=(14, 6))
#     ax2 = ax1.twinx()

#     ax1.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)', color='blue')
#     ax2.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)', color='red')

#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Feature 1 (speed)', color='blue')
#     ax2.set_ylabel('Feature 2 (angle)', color='red')

#     ax1.tick_params(axis='y', labelcolor='blue')
#     ax2.tick_params(axis='y', labelcolor='red')

#     plt.title('Features Over Time')
#     plt.legend()
#     plt.show()


# train_data, val_data, test_data = val_shift_split(3, 0)


# plotting input data
def plot_data(data_tuple):
    time_points = data_tuple[0].numpy()  
    feature_data = data_tuple[1].numpy() 
    
    plt.figure(figsize=(14, 6))

    
    plt.plot(time_points, feature_data[:, 0], label='Angle (Delta)') 
    plt.plot(time_points, feature_data[:, 1], label='frequency (f)')
    plt.plot(time_points, feature_data[:, 2], label='Voltage (V)')
    plt.plot(time_points, feature_data[:, 3], label='Power (P)')
    plt.plot(time_points, feature_data[:, 4], label='Reactive power (Q)')

    plt.title('Features Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Values')
    plt.legend()


if __name__ == "__main__":
    # Add your code here
    savefile = True
    
    data = load_data()
    plot_data(data)
    plt.show()

    # train_data, val_data, test_data = simple_split(data, 3, 0)
    # train_data, val_data, test_data = val_shift_split(data, 3, .2)

    if savefile:
        # tf.saved_model.save(data, "real_data_scuffed1")

        torch.save(data, berend_scufed)


    
    