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


def load_data_avg_duplicates(filename=laetitia_path, shift=0, start=300):
    '''
    Averages the duplicate time points rather than deleting them.
    Also fixes time points to be evenly spaced.
    '''
    # Import data
    data = pd.read_csv(filename, delimiter=',')

    # Check for duplicates
    duplicates = data.duplicated(subset=['t'])
    if duplicates.any():
        print("Duplicates found in the time axis. Averaging duplicates...")
        data = data.groupby('t', as_index=False).mean()
    else:
        print("No duplicates found in the time axis.")

    # Replace old time column with evenly spaced time series
    new_time_series = pd.Series(np.linspace(data['t'].iloc[0], data['t'].iloc[-1], len(data)))
    data['t'] = new_time_series

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
    #normalizing features with mean 0 and std 1
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


# Try fixing the time points


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

def get_batch(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, batch_dur_time=None, timestep=None, device=torch.device("cpu")):
    #maybe later improve:  when using time do math using time then convert to index to reduce rounding errors
    t_tensor, features_tensor = data_tuple

    if not timestep and (not batch_dur_idx or not batch_range_idx):
        timestep = get_timestep(t_tensor)

    if not batch_dur_idx:
        if not batch_dur_time:
            print("batch_dur_idx and batch_dur_time are both None, please specify one of them")
            return None
        batch_dur_idx = int(batch_dur_time / timestep)

    if not batch_range_idx:
        if not batch_range_time:
            print("batch_range_idx and batch_range_time are both None, please specify one of them")
            return None
        batch_range_idx = int(batch_range_time / timestep)


    s = torch.from_numpy(np.random.choice(np.arange(batch_range_idx - batch_dur_idx, dtype=np.int64), batch_size, replace=False))
    batch_t = t_tensor[:batch_dur_idx]  # (T)
    batch_y = torch.stack([features_tensor[s + i] for i in range(batch_dur_idx)], dim=0)  # (T, M, D)
    return batch_t.to(device), batch_y.to(device)

#SCUFFED VERSION of merts idea (unless i look at source code of odeint and change it)
def get_batch2(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, batch_dur_time=None, timestep=None, device=torch.device("cpu")):
    #maybe later improve:  when using time do math using time then convert to index to reduce rounding errors
    t_tensor, features_tensor = data_tuple

    if not timestep and (not batch_dur_idx or not batch_range_idx):
        timestep = get_timestep(t_tensor)

    if not batch_dur_idx:
        if not batch_dur_time:
            print("batch_dur_idx and batch_dur_time are both None, please specify one of them")
            return None
        batch_dur_idx = int(batch_dur_time / timestep)

    if not batch_range_idx:
        if not batch_range_time:
            print("batch_range_idx and batch_range_time are both None, please specify one of them")
            return None
        batch_range_idx = int(batch_range_time / timestep)


    s = torch.from_numpy(np.random.choice(np.arange(batch_range_idx - batch_dur_idx, dtype=np.int64), batch_size, replace=False))
    
    batch_t = [t_tensor[0:s[i]+batch_dur_idx] for i in range(batch_size)] # (T)
    batch_y = torch.stack([features_tensor[s + i] for i in range(batch_dur_idx)], dim=0)  # (T, M, D)
    return s, batch_t, batch_y.to(device)

#Merts idea but actually this time
def get_batch3(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, batch_dur_time=None, timestep=None, device=torch.device("cpu")):
    #maybe later improve:  when using time do math using time then convert to index to reduce rounding errors
    t_tensor, features_tensor = data_tuple

    if not timestep and (not batch_dur_idx or not batch_range_idx):
        timestep = get_timestep(t_tensor)

    if not batch_dur_idx:
        if not batch_dur_time:
            print("batch_dur_idx and batch_dur_time are both None, please specify one of them")
            return None
        batch_dur_idx = int(batch_dur_time / timestep)

    if not batch_range_idx:
        if not batch_range_time:
            print("batch_range_idx and batch_range_time are both None, please specify one of them")
            return None
        batch_range_idx = int(batch_range_time / timestep)


    s = torch.from_numpy(np.random.choice(np.arange(batch_range_idx - batch_dur_idx, dtype=np.int64), batch_size, replace=False))
    batch_t = t_tensor[:batch_range_idx]  # (T)
    batch_y = torch.stack([features_tensor[s + i] for i in range(batch_dur_idx)], dim=0)  # (T, M, D)
    return s, batch_t.to(device), batch_y.to(device)

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
    savefile = False
    
    # data = load_data()
    data = load_data_avg_duplicates()
    plot_data(data)
    plt.show()

    # train_data, val_data, test_data = simple_split(data, 3, 0)
    # train_data, val_data, test_data = val_shift_split(data, 3, .2)

    if savefile:
        # tf.saved_model.save(data, "real_data_scuffed1")

        torch.save(data, "real_data_scuffed2.pt")


    
    