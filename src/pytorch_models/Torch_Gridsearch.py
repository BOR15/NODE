import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import numpy as np
from time import perf_counter as time
import matplotlib.pyplot as plt
from logging.logsystem import saveplot, addcolumn, addlog, id
from logging.Metrics import frechet_distance

from tools.toydata_processing import get_batch, get_batch2
from tools.misc import check_cuda, tictoc
from tools.plots import *


class ODEFunc(nn.Module):
    """
    ODEFunc class represents the ordinary differential equation (ODE) function.

    Args:
        N_neurons (int): Number of neurons in the hidden layer.
        N_feat (int): Number of input features.
        device (torch.device): Device on which the computation will be performed.

    Attributes:
        net (nn.Sequential): Neural network model.
        device (torch.device): Device on which the computation will be performed.
    """

    def __init__(self, N_neurons, N_feat, device):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_feat, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, N_feat)
        )
        self.device = device
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        """
        Forward pass of the ODEFunc.

        Args:
            t (torch.Tensor): Time tensor.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        y = y.to(self.device)
        # return self.net(torch.sin(y))
        return self.net(y)


    
    
def main(num_neurons=50, num_epochs=300, learning_rate=0.01, batch_size=50, batch_dur_idx=20, batch_range_idx=500, rel_tol=1e-7, abs_tol=1e-9, val_freq=5, mert_batch=False, intermediate_pred_freq=0, live_plot=False, savemodel=False, savepredict=False):
    """
    Main function for training and evaluating a PyTorch model using ODE integration.

    Args:
        num_neurons (int): Number of neurons in the ODE function.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        rel_tol (float): Relative tolerance for the ODE solver.
        abs_tol (float): Absolute tolerance for the ODE solver.
        val_freq (int): Frequency of validation evaluation (in terms of epochs).
        intermediate_pred_freq (int): Frequency of intermediate predictions (in terms of epochs). 
            Note: for no intermediate predictions, set this to 0.
        live_plot (bool): Whether to enable live plotting of training and validation losses.
    
    Returns:
        None
    """



    #MT
    train_losses_cache = []
    train_losses = []
    val_losses = []

    # use cuda? No
    device = check_cuda(use_cuda=False)
    
    #import preprocessed data
    berend_path = r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\real_data_scuffed1.pt"
    laetitia_path = "/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/Input_Data/real_data_scuffed40h17_avg.pt"
    boris_path = "NODE/Input_Data/real_data_scuffed1.pt"

    data = torch.load("NODE/Input_Data/real_data_scuffed1.pt")
    num_feat = data[1].shape[1]

    #defining model, loss function and optimizer
    net = ODEFunc(num_neurons, num_feat, device=device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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

    #training loop
    for epoch in range(num_epochs):
        

        #training
        optimizer.zero_grad()

        # pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")
        
        
        if mert_batch: #this right now forces the code to unparaledize, which is makes it really slow but maybe i can change odeint sourcecode so it works.
            #get batch
            s, t, features = get_batch2(data, batch_size = batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            pred_y = []
            #loop through minibatches
            for i in range(batch_size):
                #doing predict
                pred_y.append(odeint(net, data[1][0], t[i], rtol=rel_tol, atol=abs_tol, method="dopri5")[-20:])
            pred_y = torch.stack(pred_y).reshape(20, 50, 5)
        else:
            #get batch
            t, features = get_batch(data, batch_size = batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            #doing predict
            pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")

        
        loss = loss_function(pred_y, features)
        loss.backward()
        optimizer.step()
        
        #save loss
        train_losses_cache.append(loss.item())
        
        #validation
        if epoch % val_freq == val_freq-1:
            with torch.no_grad():
                pred_y_val = odeint(net, data[1][0], data[0]) 
                loss_val = loss_function(pred_y_val, data[1])
            
            train_losses.append(np.mean(train_losses_cache))
            val_losses.append(loss_val.item())

            print(f"Epoch {epoch+1}: loss = {loss.item()}, val_loss = {loss_val.item()}")


            #live training vs validation plot
            if live_plot:
                line1.set_data(range(0, epoch + 1, 5), train_losses)
                line2.set_data(range(0, epoch + 1, 5), val_losses)
                ax.relim()  # Recalculate limits
                ax.autoscale_view(True,True,True)  # Autoscale
                plt.draw()
                plt.pause(0.3)  # Pause to update the plot
        
        else:
            print(f"Epoch {epoch+1}: loss = {loss.item()}")

        
        #intermediate prediction
        if intermediate_pred_freq and epoch % intermediate_pred_freq == intermediate_pred_freq-1:
            with torch.no_grad():
                predicted_intermidiate = odeint(net, data[1][0], data[0])
                evaluation_loss_intermidiate = loss_function(predicted_intermidiate, data[1]).item()
            print(f"Mean Squared Error Loss intermidiate: {evaluation_loss_intermidiate}")
            intermediate_prediction(data, predicted_intermidiate, evaluation_loss_intermidiate, num_feat, epoch)

    


    # Final predict
    with torch.no_grad():
        predicted = odeint(net, data[1][0], data[0])
        evaluation_loss = loss_function(predicted, data[1]).item()
    print(f"Mean Squared Error Loss: {evaluation_loss}")


    #Frechet distance similairity metric
    Frechet_distance = frechet_distance(net, data[1], predicted)

    id = id()

    logdict = {
        "num_neurons" : num_neurons,
        "num_epochs" : num_epochs,
        "learning_rate" : learning_rate,
        "batch_size" : batch_size,
        "batch_dur_idx" : batch_dur_idx,
        "batch_range_idx" : batch_range_idx,
        "rel_tol" : rel_tol,
        "abs_tol" : abs_tol,
        "val_freq" : val_freq,
        "mert_batch" : mert_batch,
        "loss_function" : loss_function,
        "optimizer" : optimizer,
        'frechet distance' : Frechet_distance

    }
    # saving model and predict
    if savemodel:
        torch.save(net,  f"logging/Models/{id}.pth")
    if savepredict:
        torch.save(predicted, f"logging/Predictions/{id}.pt")

    addlog('logging/log.csv', logdict)

    # Plotting 
    # TODO add saving for the plots.
        
    saveplot(plot_training(train_losses), "TrainingLoss", id)
    saveplot(plot_validation(val_losses), "ValidationLoss", id)
    saveplot(plot_actual_vs_predicted_full("true_y, pred_y, num_feat=2, toy=False, for_torch=True"), "FullPredictions", id) #TODO add info for plot. and potentially some y limits

    # plot_data(data)
    # plot_actual_vs_predicted_full(data, predicted, num_feat=num_feat)
    # plot_training_vs_validation([train_losses, val_losses], share_axis=True)

    # plt.show(block=True) #Ig we dont need this to save the graphs?

    
    




if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass
    
