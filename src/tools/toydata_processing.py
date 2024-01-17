import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 


def load_data(num_feat, filename="NODE/Input_Data/toydatafixed.csv", for_torch=True):
    # import data
    data = pd.read_csv(filename, delimiter=';')

    # check for duplicates
    duplicates = data.duplicated(subset=['t'])
    if duplicates.any():
        print("Duplicates found in the time axis.")
    else:
        print("No duplicates found in the time axis.")

    if for_torch:
        # defining tensors for Torch
        t_tensor = torch.tensor(data['t'].values, dtype=torch.float32)
        features_tensor = torch.tensor(data.iloc[:, 1:num_feat + 1].values, dtype=torch.float32)
    else:
        # defining tensors for TensorFlow
        t_tensor = tf.constant(data['t'].values, dtype=tf.float32)
        features_tensor = tf.constant(data.iloc[:, 1:num_feat + 1].values, dtype=tf.float32)


    return t_tensor, normalize_data(features_tensor, for_torch=for_torch)

def normalize_data(features_tensor, for_torch=True):
    #normalizing features between 0 and 1
    if for_torch:
        min_vals = torch.min(features_tensor, dim=0)[0]
        max_vals = torch.max(features_tensor, dim=0)[0]
    else:
        min_vals = tf.reduce_min(features_tensor, axis=0)
        max_vals = tf.reduce_max(features_tensor, axis=0)
    
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


def get_timestep(t_tensor, for_torch=True):
    if for_torch:
        timestep = torch.min(t_tensor[1:] - t_tensor[:-1])
    else:
        timestep = tf.reduce_min(t_tensor[1:] - t_tensor[:-1])
    return timestep


def simple_split(data_tuple, train_dur, val_dur=0, timestep=None, for_torch=True):
    t_tensor, features_tensor = data_tuple
    
    if not timestep:
        timestep = get_timestep(t_tensor, for_torch=for_torch)
    
    split_train = int(train_dur / timestep)
    split_val = int((val_dur + train_dur) / timestep)

    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
    test_data = (t_tensor[split_val:], features_tensor[split_val:])

    print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]}, test size: {test_data[0].shape[0]}")

    return train_data, val_data, test_data


def val_shift_split(data_tuple, train_dur, val_shift, timestep=None, for_torch=True):
    t_tensor, features_tensor = data_tuple

    if not timestep:
        timestep = get_timestep(t_tensor, for_torch=for_torch)

    #time to index
    split_train = int(train_dur / timestep)  
    shift_val = int(val_shift  / timestep)

    train_data = (t_tensor[:split_train], features_tensor[:split_train])
    val_data = (t_tensor[shift_val:split_train + shift_val], features_tensor[shift_val:split_train + shift_val])
    test_data = (t_tensor[split_train:], features_tensor[split_train:])

    print(f"training size: {train_data[0].shape[0]}, validation size: {val_data[0].shape[0]} starting at {shift_val}, test size: {test_data[0].shape[0]}")

    return train_data, val_data, test_data


def plot_data(data_tuple):
    time_points = data_tuple[0].numpy()  
    feature_data = data_tuple[1].numpy() 

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)', color='blue')
    ax2.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)', color='red')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Feature 1 (speed)', color='blue')
    ax2.set_ylabel('Feature 2 (angle)', color='red')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Features Over Time')
    plt.legend()
    plt.show()

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


# train_data, val_data, test_data = val_shift_split(3, 0)

def get_batch_tensorflow(data_tuple, batch_size, batch_range_idx=None, batch_range_time=None, batch_dur_idx=None, batch_dur_time=None, timestep=None, device="/GPU:0"):
    """This is basically the same as get_batch, but it is implemented for TensorFLow"""
    t_tensor, features_tensor = data_tuple

    if not timestep and (not batch_dur_idx or not batch_range_idx):
        timestep = get_timestep(t_tensor, for_torch=False)
    
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
    
    s = tf.convert_to_tensor(np.random.choice(np.arange(batch_range_idx - batch_dur_idx, dtype=np.int64), batch_size, replace=False))

    # creating a list of values from features_tensor (this should be the same as the torch.stack expression given in get_batch)
    x = []
    for index in s:
        for i in range(batch_dur_idx):
            y = features_tensor[index+i]
            x.append(y)
    with tf.device(device):
        batch_t = t_tensor[:batch_dur_idx]
        batch_y = tf.stack(x, axis=0)
    return batch_t, batch_y

# batch_time = 20
# batch_size = 50

# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(len(t_tensor) - batch_time-1100, dtype=np.int64), batch_size, replace=False))
#     batch_t = t_tensor[:batch_time]  # (T)
#     batch_y = torch.stack([features_tensor[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
#     return batch_t, batch_y

if __name__ == "__main__":
    # Add your code here
    savefile = False
    
    data = load_data(2, for_torch=False)
    # plot_data(data)

    # train_data, val_data, test_data = simple_split(data, 3, 0)
    # train_data, val_data, test_data = val_shift_split(data, 3, .2)

    # loaded_data = np.load('tensors.npz')
    # data = (tf.convert_to_tensor(loaded_data['t']), tf.convert_to_tensor(loaded_data['features']))

    if savefile:
        t, features = data[0].numpy(), data[1].numpy()
        np.savez('tensors.npz', t=t, features=features)

        # torch.save(data, "toydata_norm_tensorflow.pt")

    
    