import torch
import torch.nn as nn

"""
currently everything just uses MSE loss, but this is where we can define custom loss functions
definedly should try some more loss functions so to be expanded on 
"""

# MAELoss = nn.L1Loss()
# MSELoss = nn.MSELoss()

def mean_third_power_error(y_true, y_pred):
    return torch.mean(abs(y_true - y_pred) ** 3)

def mean_fourth_power_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 4)

def normalized_loss(pred, real):
    element_wise_loss = (pred - real)**2
    norm_element_wise_loss = element_wise_loss / (real.abs() + 1e-6)
    return torch.mean(norm_element_wise_loss)

def var_norm_loss(pred, real):
    loss_func = nn.MSELoss()
    var = torch.var(real)
    return loss_func(pred, real) / var


