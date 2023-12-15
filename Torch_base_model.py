import torch
import torch.nn as nn  
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from Data_preproc import simple_split, val_shift_split
import matplotlib.pyplot as plt
from time import perf_counter as time


class ODEFunc(nn.Module):
    def __init__(self, N_feat, N_neurons):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_feat, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, N_feat)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y = y
        return self.net(y)

def plot_data(data_tuple):
    time_points = data_tuple[0].numpy()  
    feature_data = data_tuple[1].numpy() 
    
    plt.figure(figsize=(14, 6))

    
    plt.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)')
    plt.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)')
    # plt.plot(time_points, feature_data[:, 2], label='Feature 3 (e_q_t)')
    # plt.plot(time_points, feature_data[:, 3], label='Feature 4 (e_q_st)')

    plt.title('Features Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Values')
    plt.legend()

    
def plot_actual_vs_predicted_full(true_y, pred_y):
    t = true_y[0].detach().numpy()
    true_y = true_y[1].detach().numpy()
    pred_y = pred_y.detach().numpy()
    

    fig, axes = plt.subplots(nrows=2, ncols=int(num_feat/2), figsize=(12, 8))
    fig.suptitle('Actual vs Predicted Features Full')

    feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, true_y[:,i], label='Actual ' + feature_names[i])
        ax.plot(t, pred_y[:,i], label='Predicted ' + feature_names[i], linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_phase_space(true_y, pred_y):
    true_y = true_y[1].detach().numpy()
    pred_y = pred_y.detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.plot(true_y[:,0], true_y[:, 1], label='Actual')
    plt.plot(pred_y[:,0], pred_y[:, 1], label='Predicted', linestyle='--')
    plt.xlabel('Speed')
    plt.ylabel('Angle')
    plt.title('Phase Space: Speed vs Angle')
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_training_vs_validation(losses):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(losses[0], label='Training Loss', color='blue')  
    ax2.plot(losses[1], label='Validation Loss', color='orange')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='blue')
    ax2.set_ylabel('Validation Loss', color='orange')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Training vs Validation Loss')
    
    ax1.legend(loc='upper left')  
    ax2.legend(loc='upper right')


if __name__ == "__main__":
    # Defining empty lists for the data
    train_losses = []
    val_losses = []
    
    # Load the saved data and split it into train, val and test
    data = torch.load("toydata_norm_0_1.pt")
    train_data, val_data, test_data = val_shift_split(data, 3, .1)

    # Defining parameters
    num_feat = data[1][1].shape[0]
    num_neurons = 50
    num_epochs = 50
    learning_rate = 0.01

    # Defining model, loss function and optimizer
    net = ODEFunc(num_feat, num_neurons)
    MSELoss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    start = time()

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_y = odeint(net, train_data[1][0], train_data[0])
        loss = MSELoss(pred_y, train_data[1])
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            pred_y = odeint(net, val_data[1][0], val_data[0])
            val_loss = MSELoss(pred_y, val_data[1])
            
        # Appending losses to lists
        print(f"Epoch: {epoch+1}, Training loss: {round(loss.item(), 5)} Validation loss: {round(val_loss.item(), 5)}")
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    print(f"Training time: {time() - start} seconds")

    # predict
    with torch.no_grad():
        predicted = odeint(net, data[1][0], data[0])
        evaluation_loss = MSELoss(predicted, data[1]).item()
    print(f"Mean Squared Error Loss: {evaluation_loss}")


    # Plotting the losses
    plot_data(data)
    plot_actual_vs_predicted_full(data, predicted)
    plot_phase_space(data, predicted)
    plot_training_vs_validation([train_losses, val_losses])
    plt.show()
