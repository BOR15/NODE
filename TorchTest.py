import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
from time import time
import matplotlib.pyplot as plt


print(f"gpu availible: {torch.cuda.is_available()}")
use_cuda = False
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using {device}")

data = pd.read_csv("toydatafixed.csv", delimiter=';')
t_tensor = torch.tensor(data['t'].values, dtype=torch.float32).to(device)
# features_tensor = torch.tensor(data.drop('t', axis=1).values, dtype=torch.float32)
features_tensor = torch.tensor(data[['speed', 'angle']].values, dtype=torch.float32).to(device)
min_vals = torch.min(features_tensor, dim=0)[0]
max_vals = torch.max(features_tensor, dim=0)[0]
features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
tensor_data = (t_tensor, features_tensor)
num_feat = features_tensor.shape[1]


def simple_split():
    split_train_dur = 2
    split_val_dur = 2

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


train_data, val_data, test_data = val_shift_split(3, .3)

print(train_data[0].shape, val_data[0].shape, test_data[0].shape, train_data[1].shape, val_data[1].shape, test_data[1].shape)



def tictoc(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time() - t1
        print(f"Time elapsed for '{func.__name__}': {t2} seconds")
        return result
    return wrapper



class ODEFunc(nn.Module):
    def __init__(self, N_neurons):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feat, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, num_feat)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y = y.to(device)
        return self.net(y**3)


#loss functions


MSELoss = nn.MSELoss()

def normalized_loss(pred, real):
    element_wise_loss = (pred - real)**2
    norm_element_wise_loss = element_wise_loss / (real.abs() + 1e-6)
    return torch.mean(norm_element_wise_loss)

def var_norm_loss(pred, real):
    loss_func = nn.MSELoss()
    var = torch.var(real)
    return loss_func(pred, real) / var


def mean_third_power_error(y_true, y_pred):
    return torch.mean(abs(y_true - y_pred) ** 3)

def mean_fourth_power_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 4)

#cross entropy loss

@tictoc
def trainmodel(data, data2, learning_rate, num_epochs, num_neurons, rel_tol=1e-7, abs_tol=1e-9, live_plot=False, intermidiate_prediction=False):
    """
    Trains a neural network model using the NODE (Neural Ordinary Differential Equations) approach.

    Args:
        data (tuple): A tuple containing the training time points (t) and features.
        data2 (tuple): A tuple containing the validation time points (val_t) and features.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of training epochs.
        num_neurons (int): The number of neurons in the ODEFunc.
        rel_tol (float): The relative tolerance for the ODE solver.
        abs_tol (float): The absolute tolerance for the ODE solver.
        live_plot (bool): Whether to enable live plotting of the training and validation losses.
        intermidiate_prediction (bool): Whether to perform intermediate predictions during training.

    Returns:
        tuple: A tuple containing the trained neural network model and a tuple of training and validation losses.
    """
    train_losses = []
    val_losses = []
    t, features = data
    val_t, val_features = data2

    net = ODEFunc(N_neurons=num_neurons).to(device)
    loss_function = MSELoss

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    if live_plot:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        line1, = ax.plot([], [], label='Training Loss')  # Line for training loss
        line2, = ax.plot([], [], label='Validation Loss')  # Line for validation loss
        ax.set_title('Training vs Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()
        plt.pause(0.1)

    def closure():
        optimizer.zero_grad()
        pred_y = odeint(net, features[0], t)
        loss = loss_function(pred_y, features)
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        
        if isinstance(optimizer, torch.optim.LBFGS):
            loss = optimizer.step(closure)
        
        else:
            optimizer.zero_grad()
            pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")
            loss = loss_function(pred_y, features)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_y_val = odeint(net, val_features[0], val_t) 
            loss_val = loss_function(pred_y_val, val_features)
        
        train_losses.append(loss.item())
        val_losses.append(loss_val.item())

        print(f"Epoch {epoch+1}: loss = {loss.item()}, val_loss = {loss_val.item()}")

        if live_plot and epoch % 5 == 4:
            line1.set_data(range(epoch + 1), train_losses)
            line2.set_data(range(epoch + 1), val_losses)
            ax.relim()  # Recalculate limits
            ax.autoscale_view(True,True,True)  # Autoscale
            plt.draw()
            plt.pause(0.2)  # Pause to update the plot

        

        if intermidiate_prediction and epoch % 30 == 29:
            intermidiate_prediction(net, epoch)
    
    return net, (train_losses, val_losses)



def intermidiate_prediction(network, epoch):
    with torch.no_grad():
        predicted_intermidiate = odeint(network, tensor_data[1][0], tensor_data[0])
    evaluation_loss_intermidiate = MSELoss(predicted_intermidiate, tensor_data[1]).item()
    print(f"Mean Squared Error Loss intermidiate: {evaluation_loss_intermidiate}")

    fig, axes = plt.subplots(nrows=2, ncols=int(num_feat/2), figsize=(12, 8))
    fig.suptitle(f'Actual vs Predicted Features, epoch = {epoch+1}, MSELoss = {evaluation_loss_intermidiate}')

    feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
    for i, ax in enumerate(axes.flatten()):
        ax.plot(tensor_data[0].cpu().numpy(), tensor_data[1][:, i].cpu().detach().numpy(), label='Actual ' + feature_names[i])
        ax.plot(tensor_data[0].cpu().numpy(), predicted_intermidiate[:, i].cpu().detach().numpy(), label='Predicted ' + feature_names[i], linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.1)





network, losses = trainmodel(train_data, val_data, 0.01, 150, 50, 1e-7, 1e-5)



plot_data = False
plot_actual_vs_predicted = False
plot_actual_vs_predicted_full = True
plot_training_vs_validation = True
plot_phase_space = True




if plot_data:
    time_points = t_tensor.numpy()  
    feature_data = features_tensor.numpy() 

    
    plt.figure(figsize=(14, 6))

    
    plt.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)')
    plt.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)')
    # plt.plot(time_points, feature_data[:, 2], label='Feature 3 (e_q_t)')
    # plt.plot(time_points, feature_data[:, 3], label='Feature 4 (e_q_st)')

    plt.title('Features Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Values')
    plt.legend()


if plot_actual_vs_predicted:

    predicted = odeint(network, val_data[1][-1], test_data[0])
    evaluation_loss = MSELoss(predicted, test_data[1]).item()
    print(f"Mean Squared Error Loss: {evaluation_loss}")

    fig, axes = plt.subplots(nrows=2, ncols=int(num_feat/2), figsize=(12, 8))
    fig.suptitle('Actual vs Predicted Features')

    feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
    for i, ax in enumerate(axes.flatten()):
        ax.plot(test_data[0].cpu().numpy(), test_data[1][:, i].cpu().detach().numpy(), label='Actual ' + feature_names[i])
        ax.plot(test_data[0].cpu().numpy(), predicted[:, i].cpu().detach().numpy(), label='Predicted ' + feature_names[i], linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if plot_actual_vs_predicted_full:
    with torch.no_grad():
        predicted_full = odeint(network, tensor_data[1][0], tensor_data[0])
    evaluation_loss_full = MSELoss(predicted_full, tensor_data[1]).item()
    print(f"Mean Squared Error Loss Full: {evaluation_loss_full}")

    fig, axes = plt.subplots(nrows=2, ncols=int(num_feat/2), figsize=(12, 8))
    fig.suptitle('Actual vs Predicted Features Full')

    feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
    for i, ax in enumerate(axes.flatten()):
        ax.plot(tensor_data[0].cpu().numpy(), tensor_data[1][:, i].cpu().detach().numpy(), label='Actual ' + feature_names[i])
        ax.plot(tensor_data[0].cpu().numpy(), predicted_full[:, i].cpu().detach().numpy(), label='Predicted ' + feature_names[i], linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if plot_training_vs_validation:
    plt.figure(figsize=(10, 6))
    plt.plot(losses[0], label='Training Loss')
    plt.plot(losses[1], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


if plot_phase_space:
    with torch.no_grad():
        predicted_full = odeint(network, tensor_data[1][0], tensor_data[0])
    evaluation_loss_full = MSELoss(predicted_full, tensor_data[1]).item()
    print(f"Mean Squared Error Loss Full: {evaluation_loss_full}")

    plt.figure(figsize=(8, 6))
    plt.plot(tensor_data[1][:, 0].cpu().detach().numpy(), tensor_data[1][:, 1].cpu().detach().numpy(), label='Actual')
    plt.plot(predicted_full[:, 0].cpu().detach().numpy(), predicted_full[:, 1].cpu().detach().numpy(), label='Predicted', linestyle='--')
    plt.xlabel('Speed')
    plt.ylabel('Angle')
    plt.title('Phase Space: Speed vs Angle')
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plt.show()


