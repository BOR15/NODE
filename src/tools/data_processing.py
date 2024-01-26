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

    print("Time tensor shape:", t_tensor.shape, "; Features tensor shape:", features_tensor.shape)
    print("Time tensor: ", t_tensor[0], "; Features tensor: ", features_tensor[0])

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
    time_aligned = time_clean - time_clean[0]
    features_clean = features_tensor[start_idx:]

    return time_aligned, features_clean


def linspace_time_series(t_tensor: torch.Tensor) -> torch.Tensor:
    '''Return new time tensor with linearly-spaced points.'''
    t_tensor_linspaced = torch.tensor(np.linspace(t_tensor[0], t_tensor[-1], len(t_tensor)))
    return t_tensor_linspaced


# def get_split_indexes(filepath, train_dur, val_dur=None):
#     '''Return indexes in time series corresponding to train (and val) time(s).'''
#     t_tensor, features = torch.load(filepath)

#     # Round times to 1 decimal point
#     train_time = round(train_dur * 10) / 10  
#     time_rounded = torch.round(t_tensor * 10) / 10

#     train_index = [index for index, value in enumerate(time_rounded) if train_time == value][0]

#     # Validation time is not used at the moment
#     if val_dur:
#         val_time = round((train_dur + val_dur) * 10) / 10
#         val_index = [index for index, value in enumerate(time_rounded) if val_time == value][0]

#     return train_index, val_index


def interpolate_features(data: tuple[torch.Tensor, torch.Tensor], num_sample_points:int = 1024, 
                         interpolation_kind:str='linear') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Linearly interpolate missing values in the features_tensor based on the time points in t_tensor.
    Sample `num_sample_points` evenly spaced points from the interpolated result.
    """
    t_tensor, features_tensor = data

    # Convert tensors to numpy arrays for interpolation
    t_np = t_tensor.numpy()
    features_np = features_tensor.numpy()

    # Create an interpolation function for each feature column
    interpolators = [interp1d(t_np, feature_column, kind=interpolation_kind, fill_value="extrapolate") for feature_column in features_np.T]

    # Interpolate missing values
    interpolated_features_np = np.array([interp(np.linspace(t_np[0], t_np[-1], num=num_sample_points)) for interp in interpolators]).T

    # Convert back to PyTorch tensors
    interpolated_features_tensor = torch.tensor(interpolated_features_np, dtype=torch.float32)
    sampled_timepoints = torch.tensor(np.linspace(t_np[0], t_np[-1], num=num_sample_points), dtype=torch.float32)

    return sampled_timepoints, interpolated_features_tensor


# def remove_spikes(t_tensor: torch.Tensor, features_tensor:torch.Tensor, spike_threshold:float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Remove the first few data points up to the first discontinuity spike after interpolation.
#     The threshold defines the maximum allowed relative change in feature values.
#     """
#     # Convert tensors to numpy arrays
#     t_np = t_tensor.numpy()
#     features_np = features_tensor.numpy()

#     # Calculate relative changes in feature values
#     relative_changes = np.abs((features_np[1:] - features_np[:-1]) / features_np[:-1])

#     # Find the index of the first spike exceeding the threshold
#     spike_idx = np.argmax(relative_changes > spike_threshold)

#     # Remove data up to the spike
#     t_tensor_no_spikes = t_tensor[spike_idx:]
#     features_tensor_no_spikes = features_tensor[spike_idx:]

#     return t_tensor_no_spikes, features_tensor_no_spikes


def normalize_data(features_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Normalizes data between 0 and 1.
    '''
    min_vals = torch.min(features_tensor, dim=0)[0]
    print("The minimum values are", min_vals)
    max_vals = torch.max(features_tensor, dim=0)[0]
    features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
    return features_tensor


def normalize_data_mean_0(features_tensor: torch.Tensor, for_torch=True) -> torch.Tensor:
    '''Normalizes features with mean 0 and std 1.'''
    if for_torch:
        mean_vals = torch.mean(features_tensor, dim=0)
        std_vals = torch.std(features_tensor, dim=0)
    else:
        mean_vals = tf.reduce_mean(features_tensor, axis=0)
        std_vals = tf.math.reduce_std(features_tensor, axis=0)
    
    features_tensor = (features_tensor - mean_vals) / std_vals
    return features_tensor


def get_timestep(t_tensor: torch.Tensor) -> torch.Tensor:
    '''Returns the minimum timestep between each point.'''
    timestep = torch.min(t_tensor[1:] - t_tensor[:-1])
    return timestep


def random_sampling(data: tuple[torch.Tensor, torch.Tensor], num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
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
#     Splitting the data the same way as the function above but taking validation shift instead of duration.
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


def get_batch(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, 
              batch_dur_time=None, timestep=None, device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
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

    s = torch.from_numpy(np.random.choice(np.arange(batch_range_idx - batch_dur_idx, dtype=np.int64), batch_size, replace=True))
    batch_t = t_tensor[:batch_dur_idx]  # (T)
    batch_y = torch.stack([features_tensor[s + i] for i in range(batch_dur_idx)], dim=0)  # (T, M, D)
    return batch_t.to(device), batch_y.to(device)


#SCUFFED VERSION of merts idea (unless i look at source code of odeint and change it)
def get_batch2(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, 
               batch_dur_time=None, timestep=None, device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
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
def get_batch3(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, 
               batch_dur_time=None, timestep=None, device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
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


def plot_interpolated_data(data_tuple: tuple[torch.Tensor, torch.Tensor]) -> None:
    '''Plot the interpolated data over time.'''
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


def save_clean_raw_data(filepath: str, shift: int, start: int, file_suffix: str, normalization=False) -> str:
    '''
    Removes bad start of signal and saves as file.
    If normalization=True, then also save a version with normalized data and another with standardized data.
    '''
    full_filename = "clean_raw_data_" + file_suffix + ".pt"

    # Load the data and remove duplicates
    data = load_data(filepath, shift)

    # Cut the start of the data
    data_clean = clean_start(data, start)

    t_tensor, features_tensor = data_clean

    if normalization:
        mean0_filename = "clean_mean0_data_" + file_suffix + ".pt"
        normalized_filename = "clean_normalized_data_" + file_suffix + ".pt"

        torch.save((t_tensor, normalize_data_mean_0(features_tensor)), mean0_filename)
        torch.save((t_tensor, normalize_data(features_tensor)), normalized_filename)

    # save the data
    torch.save((t_tensor, features_tensor), full_filename)

    return full_filename


def save_interpolated_data(filepath: str, num_samples: int, file_suffix: str, interpolation_kind: str) -> None:
    '''
    Interpolates features of data with given number of samples num_samples and creates two files:
    one with normalized data and the other with standardized data.
    '''
    mean0_filename = "mean0_interpolated_" + file_suffix + "_" + str(num_samples) + "_samples.pt"
    normalized_filename =  "normalized_interpolated_" + file_suffix + "_" + str(num_samples) + "_samples.pt"

    data = torch.load(filepath)

    # Interpolate features
    t_tensor, features_tensor = interpolate_features(data, num_samples, interpolation_kind)

    # Normalize features
    features_tensor_mean0 = normalize_data_mean_0(features_tensor)
    features_tensor_normalized = normalize_data(features_tensor)

    # save the data
    torch.save((t_tensor, features_tensor_mean0), mean0_filename)
    torch.save((t_tensor, features_tensor_normalized), normalized_filename)


def save_stretched_time_data(filepath: str, shift: int, start: int, file_suffix: str, 
                             normalization: bool = False, downsampling=None) -> None:
    '''
    Creates time tensor as linespaced tensor of same length as original time tensor. 
    Removes bad start of signal and saves time tensor and features tensor as file.
    If normalization=True, then save a version with normalized data and another with standardized data.
    '''
    full_filename = "stretched_data_" + file_suffix + ".pt"

    # Load the data and remove duplicates
    data = load_data(filepath, shift)

    # Cut the start of the data
    data_clean = clean_start(data, start)

    t_tensor, features_tensor = data_clean
    t_tensor_linspaced = torch.tensor(np.linspace(t_tensor[0], t_tensor[-1], len(t_tensor)))

    if normalization:
        mean0_filename = "stretched_mean0_data_" + file_suffix + ".pt"
        normalized_filename = "stretched_normalized_data_" + file_suffix + ".pt"

        torch.save((t_tensor_linspaced, normalize_data_mean_0(features_tensor)), mean0_filename)
        torch.save((t_tensor_linspaced, normalize_data(features_tensor)), normalized_filename)

    # save the unnormalized data
    # torch.save((t_tensor_linspaced, features_tensor), full_filename)
    # return full_filename


g1_start = 179  # 15h23 sw_g1
g2_start = 177  # 15h23 ne_g2
g8_start = 181  # 15h23 sw_g8

g1_shift = 3
g2_shift = 0
g8_shift = 4

shifts = [g1_shift, g2_shift, g8_shift]
starts = [g1_start, g2_start, g8_start]
suffixes = ['g1', 'g2', 'g8']

shortest_data_len = 806  # len of data
interpolation_type = 'quadratic'
num_samples_interpolation = [100, 400, 1200]

data_15h23_path = r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\Raw_Data\Dynamics15h23.csv"
raw_path_root = r"C:\Users\Mieke\Documents\GitHub\NODE"


def main(data_path: str, shifts: list[int], starts: list[int], suffixes: list[str], path_root: str, interpolation: str, num_samples_interpolation: list[int]) -> None:
    '''
    This function creates all the necessary input_data files. 
    '''
    zipped_arguments = zip(shifts, starts, suffixes)

    raw_file_names = []
    for shift, start, suffix in zipped_arguments:
        save_stretched_time_data(data_path, shift, start, suffix, normalization=True)
        raw_filename = save_clean_raw_data(data_path, shift, start, suffix, normalization=True)
        raw_file_names.append(raw_filename)
    
    raw_file_paths = []
    for file_name in raw_file_names:
        raw_file_paths.append(path_root + file_name)
    
    for path in raw_file_paths:
        for num in num_samples_interpolation:
            for suffix in suffixes:
                save_interpolated_data(path, num, suffix, interpolation)


if __name__ == "__main__":
    savefile = True
    
    if savefile:
        main(data_15h23_path, shifts, starts, suffixes, raw_path_root, interpolation_type, num_samples_interpolation)


# if __name__ == "__main__":
#     '''
#     Save file of raw data with only dropping of duplicates and cutting of start. 
#     '''
#     savefile = False

#     # Load the data and remove duplicates
#     data = load_data(data_15h23_path, g1_shift)

#     # Cut the start of the data
#     data_clean = clean_start(data, g1_start)

#     t_tensor, features_tensor = data_clean

#     if savefile:
#         torch.save((t_tensor, features_tensor), "clean_raw_data_g1.pt")


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
#         torch.save((t_tensor, features_tensor_normalized), "real_data_.pt")

    
# if __name__ == "__main__":
#     '''
#     Save file of interpolated preprocessed data, taking clean_raw_data as input. 
#     '''
#     savefile = False
#     data = torch.load(filepath)

#     # Interpolation for 500, 1500, and 3000 points
#     t_tensor, features_tensor = interpolate_features(data, 400)
#     # t_tensor, features_tensor = interpolate_features(data, 1600)
#     # t_tensor, features_tensor = interpolate_features(data, 2400)

#     # Normalize features
#     features_tensor_normalized = normalize_data(features_tensor)
#     features_tensor_normalized = normalize_data_mean_0(features_tensor)

#     # # Plot the data
#     # plot_data((t_tensor, features_tensor))
#     # plt.show()

#     if savefile:
#         torch.save((t_tensor, features_tensor_normalized), "real_data_.pt")