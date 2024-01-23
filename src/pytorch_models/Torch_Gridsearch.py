import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import numpy as np
from time import perf_counter as time
import matplotlib.pyplot as plt
from tools.logsystem import saveplot, addcolumn, addlog, logid
from tools.Metrics import frechet_distance

from tools.data_processing import get_batch, get_batch2, get_batch3
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


    
    
def main(num_neurons=50, num_epochs=300, epochs=[200, 250], 
         learning_rate=0.01, loss_coefficient=1,
         batch_size=50, batch_dur_idx=20, batch_range_idx=500, 
         rel_tol=1e-7, abs_tol=1e-9, val_freq=5, 
         lmbda=5e-3, regu=None,
         mert_batch_scuffed=False, mert_batch=False,
         intermediate_pred_freq=0, live_intermediate_pred=False, live_plot=False, 
         savemodel=False, savepredict=False):
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
    scores = []

    def logging():
        # Final predict
        with torch.no_grad():
            predicted = odeint(net, data[1][0], data[0])
            evaluation_loss = loss_function(predicted, data[1]).item()
        print(f"Mean Squared Error Loss: {evaluation_loss}")


        #Frechet distance similairity metric
        Frechet_distance = frechet_distance(data[1], predicted) #TODO fix this
        

        logi = logid()

        logdict = {
            "logid" : logi,
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

        # addlog('logging/log.csv', logdict) #TODO FIX THIS 

        # # Plotting   
        # saveplot(plot_training_vs_validation([train_losses, val_losses], sample_freq=val_freq, two_plots=True), "Losses", id)
        # saveplot(plot_actual_vs_predicted_full(data, predicted, num_feat=num_feat, toy=False, for_torch=True), "FullPredictions", id) #TODO add args for subtitle

        scores = [1, 2, 3]  ##TODO add more scores here
        return scores
        


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

    data = torch.load("Input_Data/real_data_scuffed1.pt")  #this is the actual correct path for final submission (i think)
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
        
        
        if mert_batch_scuffed: #this right now forces the code to unparaledize, which is makes it really slow but maybe i can change odeint sourcecode so it works.
            #get batch
            s, t, features = get_batch2(data, batch_size = batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            pred_y = []
            #loop through minibatches
            for i in range(batch_size):
                #doing predict
                pred_y.append(odeint(net, data[1][0], t[i], rtol=rel_tol, atol=abs_tol, method="dopri5")[-20:])
            pred_y = torch.stack(pred_y).reshape(20, 50, 5)
        elif mert_batch:
            #get batch
            s, t, features = get_batch3(data, batch_size = batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            #doing predict
            pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")

            pred_y_cut = torch.zeros_like(pred_y)[:20,:,:]
            
            # i think this is slower then the advanced indexing but im not sure
            # for i in range(batch_size):
            #     pred_y_cut[:,i,:] = pred_y[:,i,:][s[i]:s[i]+batch_dur_idx]
            # pred_y = pred_y_cut

            range_tensor = torch.arange(0, batch_dur_idx, device=device)
            index_tensor = s[:, None] + range_tensor[None, :]
            pred_y_cut = pred_y.gather(0, index_tensor[:, :, None].expand(-1, -1, pred_y.size(2)))
            pred_y = pred_y_cut.transpose(0, 1)
            

        else:
            #get batch
            t, features = get_batch(data, batch_size = batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            #doing predict
            pred_y = odeint(net, features[0], t, rtol=rel_tol, atol=abs_tol, method="dopri5")

        
        loss = loss_coefficient * loss_function(pred_y, features)
        
        # regularisation L1 and L2 implementation
        l1, l2 = 0, 0
        if regu == 'L2':
            for p in net.parameters():
                l2 = l2 + torch.pow(p,2).sum()
            loss = loss + lmbda * l2
          
        if regu == 'L1':
            for p in net.parameters():
                l1 += torch.sum(torch.abs(p))
            loss = loss + lmbda * l1 
        
        
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
            plot_actual_vs_predicted_full(data, predicted_intermidiate, num_feat=num_feat, info=(epoch, evaluation_loss_intermidiate))
            if live_intermediate_pred:
                plt.show(block=False)
                plt.pause(0.1)


        #logging at non final epochs
        if epoch+1 in epochs:
            scores.append(logging())

    
    scores.append(logging())
    scores = np.array(scores)  #dim 0 is epochs, dim 1 is scores
    # average them? take the highest? then return: for now ill average
    scores = np.mean(scores, axis=0)
    return scores

    


    # # Final predict
    # with torch.no_grad():
    #     predicted = odeint(net, data[1][0], data[0])
    #     evaluation_loss = loss_function(predicted, data[1]).item()
    # print(f"Mean Squared Error Loss: {evaluation_loss}")


    # #Frechet distance similairity metric
    # Frechet_distance = frechet_distance(net, data[1], predicted)

    # logid = logid()

    # logdict = {
    #     "logid" : logid,
    #     "num_neurons" : num_neurons,
    #     "num_epochs" : num_epochs,
    #     "learning_rate" : learning_rate,
    #     "batch_size" : batch_size,
    #     "batch_dur_idx" : batch_dur_idx,
    #     "batch_range_idx" : batch_range_idx,
    #     "rel_tol" : rel_tol,
    #     "abs_tol" : abs_tol,
    #     "val_freq" : val_freq,
    #     "mert_batch" : mert_batch,
    #     "loss_function" : loss_function,
    #     "optimizer" : optimizer,
    #     'frechet distance' : Frechet_distance

    # }
    # # saving model and predict
    # if savemodel:
    #     torch.save(net,  f"logging/Models/{id}.pth")
    # if savepredict:
    #     torch.save(predicted, f"logging/Predictions/{id}.pt")

    # addlog('logging/log.csv', logdict)

    # # Plotting 
    # # TODO add saving for the plots.
        

    # saveplot(plot_training_vs_validation([train_losses, val_losses], sample_freq="?", two_plots=True), "Losses", id)
    # saveplot(plot_actual_vs_predicted_full(data, predicted, num_feat=num_feat, toy=False, for_torch=True), "FullPredictions", id)


    # plot_data(data)
    # plot_actual_vs_predicted_full(data, predicted, num_feat=num_feat)
    # plot_training_vs_validation([train_losses, val_losses], share_axis=True)

    # plt.show(block=True) #Ig we dont need this to save the graphs?

    
    
if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass
    
