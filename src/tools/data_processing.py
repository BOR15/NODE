import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

from scipy.interpolate import interp1d

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


def load_data(filename, shift=0):
    '''
    Loads the data from a CSV file.
    Drops duplicates in the time axis, keeping the last one only. 
    Returns two tensors: time and features.
    '''
    # Import data
    data = pd.read_csv(filename, delimiter=',')

    # Check for duplicates in the time t. average these duplicates to have a clean and non-redundant dataset.
    duplicates = data.duplicated(subset=['t'])
    if duplicates.any():
        print("Duplicates found in the time axis. Removing duplicates...")
        data = data.drop_duplicates(subset=['t'], keep='last')
    else:
        print("No duplicates found in the time axis.")

    t_tensor = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)
    features_tensor = torch.tensor(data.iloc[:, 2+shift : 27+shift : 5].values, dtype=torch.float32)

    print(features_tensor.shape, t_tensor.shape)
    print(features_tensor[0], t_tensor[0])

    return t_tensor, features_tensor


# def load_data_avg_duplicates(filename, shift=0, start=300):
#     '''
#     Loads the data from a CSV file. 
#     Replaces duplicate time points by their average value.
#     Normalizes the features with mean 0 and std 1.
#     Returns two tensors: time and features.
#     '''
#     # Import data
#     data = pd.read_csv(filename, delimiter=',')

#     # Check for duplicates
#     duplicates = data.duplicated(subset=['t'])
#     if duplicates.any():
#         print("Duplicates found in the time axis. Averaging duplicates...")
#         data = data.groupby('t', as_index=False).mean()
#     else:
#         print("No duplicates found in the time axis.")

#     # Defining tensors
#     t_tensor = torch.tensor(data.iloc[start:, 1].values, dtype=torch.float32)
#     features_tensor = torch.tensor(data.iloc[start:, 2+shift:27+shift:5].values, dtype=torch.float32)
#     print(features_tensor.shape, t_tensor.shape)
#     print(features_tensor[0], t_tensor[0])

#     features_tensor = normalize_data_mean_0(features_tensor)
#     return t_tensor, features_tensor


def clean_start(data: tuple[torch.Tensor, torch.Tensor], start_idx: int):
    '''Cuts off start of signal up to given index and realigns time tensor from 0.'''
    t_tensor, features_tensor = data

    time_clean = t_tensor[start_idx:]
    time_clean -= time_clean[0]
    features_clean = features_tensor[start_idx:]

    return time_clean, features_clean


def linspace_time_series(t_tensor):
    '''
    Output new tensor with evenly-spaced time-points.
    '''
    t_tensor_linspaced = torch.tensor(np.linspace(t_tensor[0], t_tensor[-1], len(t_tensor)))
    return t_tensor_linspaced


def get_split_indexes(filepath, train_dur, val_dur=None):
    '''Return indexes in time series corresponding to train (and val) time(s).'''
    t_tensor, features = torch.load(filepath)

    # Round times to 1 decimal point
    train_time = round(train_dur * 10) / 10  
    time_rounded = torch.round(t_tensor * 10) / 10

    train_index = [index for index, value in enumerate(time_rounded) if train_time == value][0]

    # Validation time is not used at the moment
    if val_dur:
        val_time = round((train_dur + val_dur) * 10) / 10
        val_index = [index for index, value in enumerate(time_rounded) if val_time == value][0]

    return train_index, val_index


def interpolate_features(data, num_sample_points=1024):
    """
    Linearly interpolate missing values in the features_tensor based on the time points in t_tensor.
    Sample `num_sample_points` evenly spaced points from the interpolated result.
    """
    t_tensor, features_tensor = data

    # Convert tensors to numpy arrays for interpolation
    t_np = t_tensor.numpy()
    features_np = features_tensor.numpy()

    # Create an interpolation function for each feature column
    interpolators = [interp1d(t_np, feature_column, kind='linear', fill_value="extrapolate") for feature_column in features_np.T]

    # Interpolate missing values
    interpolated_features_np = np.array([interp(np.linspace(t_np[0], t_np[-1], num=num_sample_points)) for interp in interpolators]).T

    # Convert back to PyTorch tensors
    interpolated_features_tensor = torch.tensor(interpolated_features_np, dtype=torch.float32)
    sampled_timepoints = torch.tensor(np.linspace(t_np[0], t_np[-1], num=num_sample_points), dtype=torch.float32)

    return sampled_timepoints, interpolated_features_tensor


def remove_spikes(t_tensor, features_tensor, spike_threshold=0.1):
    """
    Remove the first few data points up to the first discontinuity spike after interpolation.
    The threshold defines the maximum allowed relative change in feature values.
    """
    # Convert tensors to numpy arrays
    t_np = t_tensor.numpy()
    features_np = features_tensor.numpy()

    # Calculate relative changes in feature values
    relative_changes = np.abs((features_np[1:] - features_np[:-1]) / features_np[:-1])

    # Find the index of the first spike exceeding the threshold
    spike_idx = np.argmax(relative_changes > spike_threshold)

    # Remove data up to the spike
    t_tensor_no_spikes = t_tensor[spike_idx:]
    features_tensor_no_spikes = features_tensor[spike_idx:]

    return t_tensor_no_spikes, features_tensor_no_spikes

# #removes first few time points that are the same after interpolation
# def remove_redundant_data(t_tensor, features_tensor):
#     # Convert tensors to numpy arrays
#     t_np = t_tensor.numpy()
#     features_np = features_tensor.numpy()

#     # Find the index of the first non-redundant time point
#     non_redundant_start_idx = 0
#     for i in range(1, len(t_np)):
#         if not np.allclose(features_np[i], features_np[i - 1]):
#             non_redundant_start_idx = i
#             break

#     # Remove redundant data
#     t_tensor_non_redundant = t_tensor[non_redundant_start_idx:]
#     features_tensor_non_redundant = features_tensor[non_redundant_start_idx:]

#     return t_tensor_non_redundant, features_tensor_non_redundant


def normalize_data(features_tensor):
    '''
    Normalizes features between 0 and 1.
    It has been found that normalising the data between 0 and 1 is the best for neural networks.
    '''
    min_vals = torch.min(features_tensor, dim=0)[0]
    print(min_vals)
    max_vals = torch.max(features_tensor, dim=0)[0]
    features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
    return features_tensor


def normalize_data_mean_0(features_tensor, for_torch=True):
    '''Normalizes features with mean 0 and std 1.'''
    if for_torch:
        mean_vals = torch.mean(features_tensor, dim=0)
        std_vals = torch.std(features_tensor, dim=0)
    else:
        mean_vals = tf.reduce_mean(features_tensor, axis=0)
        std_vals = tf.math.reduce_std(features_tensor, axis=0)
    
    features_tensor = (features_tensor - mean_vals) / std_vals
    return features_tensor


def get_timestep(t_tensor):
    '''Returns the minimum timestep between each point.'''
    timestep = torch.min(t_tensor[1:] - t_tensor[:-1])
    return timestep


def random_sampling(data: tuple[torch.Tensor, torch.Tensor], num_samples: int):
    '''Sample given number of samples uniformly in data.'''
    t_tensor, features_tensor = data

    if num_samples > len(t_tensor):
        raise ValueError("The number of samples cannot be larger than the number of datapoints.")
    
    # Generate random indices
    indices = torch.randperm(len(t_tensor))[:num_samples]

    # Sample from the tensors
    sampled_t_tensor = t_tensor[indices]
    sampled_features_tensor = features_tensor[indices]

    return sampled_t_tensor, sampled_features_tensor


# def simple_split(data_tuple, train_dur, val_dur=0, timestep=None):
#     '''
#     Create a data split based on smallest timestep: number of datapoints in training and validation is simply 
#     training/validation time divided by smallest timestep.
#     This method won't work if the sampling on time is not evenly spaced through the entire data. 
#     '''
#     t_tensor, features_tensor = data_tuple
    
#     if not timestep:
#         timestep = get_timestep(t_tensor)
    
#     split_train = int(train_dur / timestep)
#     split_val = int((val_dur + train_dur) / timestep)

#     train_data = (t_tensor[:split_train], features_tensor[:split_train])
#     val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
#     test_data = (t_tensor[split_val:], features_tensor[split_val:])

#     print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]}, test size: {test_data[0].shape[0]}")

#     return train_data, val_data, test_data


# def val_shift_split(data_tuple, train_dur, val_shift, timestep=None):
#     '''
#     Splitting the data the same way as the function above but slightly different but idk why.
#     '''
#     t_tensor, features_tensor = data_tuple

#     if not timestep:
#         timestep = get_timestep(t_tensor)

#     #time to index
#     split_train = int(train_dur / timestep)  
#     shift_val = int(val_shift  / timestep)

#     train_data = (t_tensor[:split_train], features_tensor[:split_train])
#     val_data = (t_tensor[shift_val:split_train + shift_val], features_tensor[shift_val:split_train + shift_val])
#     test_data = (t_tensor[split_train:], features_tensor[split_train:])

#     print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]} starting at {shift_val}, test size: {test_data[0].shape[0]}")

#     return train_data, val_data, test_data


# # This function is not used at the moment
# def data_split(filepath, train_dur, val_dur):
#     '''
#     Splits the data based on measured timepoints rather than number of data points. 
#     Uses original time series from data to make split.
#     How to deal with linspaced time series?
#     '''
#     t_tensor, features_tensor = torch.load(filepath)

#     #Round times to 1 decimal point
#     train_time = round(train_dur * 10) / 10  
#     val_time = round((train_dur + val_dur) * 10) / 10
#     time_rounded = torch.round(t_tensor * 10) / 10  

#     train_idx = [index for index, value in enumerate(time_rounded) if train_time == value][0]
#     val_idx = [index for index, value in enumerate(time_rounded) if val_time == value][0]

#     train_data = (t_tensor[:train_idx], features_tensor[:train_idx])
#     val_data = (t_tensor[train_idx:val_idx], features_tensor[train_idx:val_idx])
#     test_data = (t_tensor[val_idx:], features_tensor[val_idx:])

#     return train_data, val_data, test_data


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

#plot interpolated data
def plot_interpolated_data(data_tuple):
    t_tensor, features_tensor = interpolate_features(data_tuple[0], data_tuple[1])
    plot_data((t_tensor, features_tensor))


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


# if __name__ == "__main__":
#     # Add your code here
#     savefile = False
    
#     # # data = load_data()
#     # #plot_data(data)

#     data = load_data_avg_duplicates()
#     plot_interpolated_data(data)
#     plt.show()

#     #previous
#     # data = load_data_avg_duplicates()
#     # t_tensor, features_tensor = interpolate_features(data[0], data[1])
#     # # Remove redundant data
#     # t_tensor_non_redundant, features_tensor_non_redundant = remove_redundant_data(t_tensor, features_tensor)
#     # # Plot the data after removing redundant points
#     # plot_data((t_tensor_non_redundant, features_tensor_non_redundant))
#     # plt.show()

#     # train_data, val_data, test_data = simple_split(data, 3, 0)
#     # train_data, val_data, test_data = val_shift_split(data, 3, .2)

#     if savefile:
#         # tf.saved_model.save(data, "real_data_scuffed1")

#         #previous
#         #torch.save((t_tensor_non_redundant, features_tensor_non_redundant), "real_data_scuffed2_non_redundant.pt")

#         torch.save(data, "real_data_scuffed2.pt")


g1_start = 179  # 15h23 sw_g1
g2_start = 177  # 15h23 ne_g2
g8_start = 181  # 15h23 sw_g8

shortest_data_len = 806  # len of data

g1_shift = 3
g2_shift = 0
g8_shift = 4

shifts = [g1_shift, g2_shift, g8_shift]
starts = [g1_start, g2_start, g8_start]
suffixes = ['g1', 'g2', 'g8']

data_15h23_path = "/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/Input_Data/Raw_Data/Dynamics15h23.csv"
raw_files = []
processed_files = []


def save_clean_raw_data(filepath: str, shift: int, start: int, file_suffix: str):
    full_filename = "clean_raw_data_" + file_suffix + ".pt"
    raw_files.append(full_filename)

    # Load the data and remove duplicates
    data = load_data(filepath, shift)

    # Cut the start of the data
    data_clean = clean_start(data, start)

    t_tensor, features_tensor = data_clean

    # save the data
    torch.save((t_tensor, features_tensor), full_filename)

    return raw_files


def save_interpolated_data(filepath: str, num_samples: int, file_suffix: str):
    mean0_filename = "mean0_interpolated_data_" + file_suffix + ".pt"
    normalized_filename =  "normalized_interpolated_data_" + file_suffix + ".pt"

    data = torch.load(filepath)

    # Interpolate features
    t_tensor, features_tensor = interpolate_features(data, num_samples)

    # Normalize features
    features_tensor_normalized = normalize_data(features_tensor)
    features_tensor_normalized = normalize_data_mean_0(features_tensor)

    # save the data
    torch.save((t_tensor, features_tensor_normalized), mean0_filename)
    torch.save((t_tensor, features_tensor_normalized), normalized_filename)


if __name__ == "__main__":
    '''
    Save file of raw data with only dropping of duplicates and cutting of start. 
    '''
    savefile = True

    # Load the data and remove duplicates
    data = load_data(data_15h23_path, g1_shift)

    # Cut the start of the data
    data_clean = clean_start(data, g1_start)

    t_tensor, features_tensor = data_clean

    if savefile:
        torch.save((t_tensor, features_tensor), "clean_raw_data_g1.pt")


# if __name__ == "__main__":
#     '''
#     Save file of randomly sampled preprocessed data, taking clean_raw_data as input. 
#     '''
#     data = torch.load(filepath)

#     # Random sampling
#     t_tensor, features_tensor = random_sampling(data, data_len)

#     # Normalize features
#     features_tensor_normalized = normalize_data(features_tensor)
#     features_tensor_normalized = normalize_data_mean_0(features_tensor)

#     # # Plot the data
#     # plot_data((t_tensor, features_tensor))
#     # plt.show()

#     if savefile:
#         # Save the data without spikes to a new file
#         torch.save((t_tensor, features_tensor_normalized), "real_data_.pt")

    
if __name__ == "__main__":
    '''
    Save file of interpolated preprocessed data, taking clean_raw_data as input. 
    '''
    data = torch.load(filepath)

    # Interpolation for 500, 1500, and 3000 points
    t_tensor, features_tensor = interpolate_features(data, 400)
    # t_tensor, features_tensor = interpolate_features(data, 1600)
    # t_tensor, features_tensor = interpolate_features(data, 2400)

    # Normalize features
    features_tensor_normalized = normalize_data(features_tensor)
    features_tensor_normalized = normalize_data_mean_0(features_tensor)

    # # Plot the data
    # plot_data((t_tensor, features_tensor))
    # plt.show()

    if savefile:
        # Save the data without spikes to a new file
        torch.save((t_tensor, features_tensor_normalized), "real_data_scuffed2_no_spikes.pt")
