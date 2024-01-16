import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import numpy as np
from time import perf_counter as time
import matplotlib.pyplot as plt

from tools.toydata_processing import get_batch
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


    
    
def main(num_neurons=50, num_epochs=300, learning_rate=0.01, rel_tol=1e-7, abs_tol=1e-9, val_freq=5, intermediate_pred_freq=0, live_plot=False):
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
    # data = torch.load("NODE/Input_Data/toydata_norm_0_1.pt")
    berend_path = r"C:\Users\Mieke\Documents\GitHub\NODE\Input_Data\real_data_scuffed1.pt"
    laetitia_path = "/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/Input_Data/real_data_scuffed1.pt"
    boris_path = "NODE/Input_Data/real_data_scuffed1.pt"

    data = torch.load(laetitia_path)
    num_feat = data[1].shape[1]

    #defining model, loss function and optimizer
    net = ODEFunc(50, num_feat, device=device)
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
        #get batch
        t, features = get_batch(data, batch_size = 50, batch_dur_idx = 20, batch_range_idx=500, device=device)

        #training
        optimizer.zero_grad()
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


    # Plotting 
    plot_data(data)
    plot_actual_vs_predicted_full(data, predicted, num_feat=num_feat)
    # plot_phase_space(data, predicted)
    plot_training_vs_validation([train_losses, val_losses], share_axis=True)
    plt.show(block=True)




if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass
    


# old shit
    

# @tictoc
# def trainmodel(data, data2, learning_rate, num_epochs, num_neurons, rel_tol=1e-7, abs_tol=1e-9, live_plot=False, intermidiate_pred=False, use_batches=False):
#     """
#     Trains a neural network model using the NODE (Neural Ordinary Differential Equations) approach.

#     Args:
#         data (tuple): A tuple containing the training time points (t) and features.
#         data2 (tuple): A tuple containing the validation time points (val_t) and features.
#         learning_rate (float): The learning rate for the optimizer.
#         num_epochs (int): The number of training epochs.
#         num_neurons (int): The number of neurons in the ODEFunc.
#         rel_tol (float): The relative tolerance for the ODE solver.
#         abs_tol (float): The absolute tolerance for the ODE solver.
#         live_plot (bool): Whether to enable live plotting of the training and validation losses.
#         intermidiate_prediction (bool): Whether to perform intermediate predictions during training.

#     Returns:
#         tuple: A tuple containing the trained neural network model and a tuple of training and validation losses.
#     """
#     train_losses_cache = []
#     train_losses = []
#     val_losses = []
#     t, features = data
#     val_t, val_features = data2

#     net = ODEFunc(N_neurons=num_neurons).to(device)
#     loss_function = MSELoss

#     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#     # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    
#     if live_plot:
#         plt.ion()  # Turn on interactive mode
#         fig, ax = plt.subplots(figsize=(10, 6))
#         line1, = ax.plot([], [], label='Training Loss')  # Line for training loss
#         line2, = ax.plot([], [], label='Validation Loss')  # Line for validation loss
#         ax.set_title('Training vs Validation Loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.legend()
#         plt.show()
#         plt.pause(0.1)

#     def closure():
#         optimizer.zero_grad()
#         pred_y = odeint(net, features[0], t)
#         loss = loss_function(pred_y, features)
#         loss.backward()
#         return loss

#     for epoch in range(num_epochs):
#         #get batch
#         if use_batches:
#             t, features = get_batch(tensor_data, batch_size = 50, batch_dur_idx = 20, batch_range_idx=200)

#         #training
#         if isinstance(optimizer, torch.optim.LBFGS): #LBFGS needs a closure function but LBFGS is prob not the best optimizer for this
#             loss = optimizer.step(closure)
        
#         else:
#             optimizer.zero_grad()
#             pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")
#             loss = loss_function(pred_y, features)
#             loss.backward()
#             optimizer.step()
        
#         train_losses_cache.append(loss.item())
#         print(f"Epoch {epoch+1}: loss = {loss.item()}")
        

#         #validation
#         if epoch % 5 == 4:
#             with torch.no_grad():
#                 pred_y_val = odeint(net, val_features[0], val_t) 
#                 loss_val = loss_function(pred_y_val, val_features)
            
#             train_losses.append(np.mean(train_losses_cache))
#             val_losses.append(loss_val.item())

#             print(f"Epoch {epoch+1}: val_loss = {loss_val.item()}")


#             #live training vs validation plot
#             if live_plot:
#                 line1.set_data(range(0, epoch + 1, 5), train_losses)
#                 line2.set_data(range(0, epoch + 1, 5), val_losses)
#                 ax.relim()  # Recalculate limits
#                 ax.autoscale_view(True,True,True)  # Autoscale
#                 plt.draw()
#                 plt.pause(0.4)  # Pause to update the plot

        
#         #intermidiate prediction
#         if intermidiate_pred and epoch % 100 == 99:
#             intermidiate_prediction(net, epoch)
    
#     return net, (train_losses, val_losses)

