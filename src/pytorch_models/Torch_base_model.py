import torch
import torch.nn as nn  
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
from time import perf_counter as time

from tools.toydata_processing import val_shift_split
from tools.plots import *

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



def main(num_neurons=50, num_epochs=50, learning_rate=0.01, train_duration=1.5, val_shift=0.1):
    """
    This function trains a PyTorch model using the given parameters and data.
    
    Args:
        num_neurons (int): Number of neurons in the model (default: 50).
        num_epochs (int): Number of training epochs (default: 50).
        learning_rate (float): Learning rate for the optimizer (default: 0.01).
        train_duration (float): Duration of the training data (default: 1.5).
        val_shift (float): Shift for the validation data (default: 0.1).
    """

    # Defining empty lists for the data
    train_losses = []
    val_losses = []
    
    berend_dataPath1= r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\real_data_scuffed1.pt"
    boris_dataPath1 = "NODE/Input_Data/toydata_norm_0_1.pt"

    # Load the saved data and split it into train, val and test
    data = torch.load(berend_dataPath1)
    train_data, val_data, test_data = val_shift_split(data, train_dur=train_duration, val_shift=val_shift)

    # Defining parameters
    num_feat = data[1][1].shape[0]
    # num_neurons = 5
    # num_epochs = 5
    # learning_rate = 0.01

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
    plot_training_vs_validation([train_losses, val_losses], share_axis=True)
    plt.show()


if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass