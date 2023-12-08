import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
from time import time
import matplotlib.pyplot as plt

data = pd.read_csv("toydatafixed.csv", delimiter=';')
t_tensor = torch.tensor(data['t'].values, dtype=torch.float32)
features_tensor = torch.tensor(data.drop('t', axis=1).values, dtype=torch.float32)
min_vals = torch.min(features_tensor, dim=0)[0]
max_vals = torch.max(features_tensor, dim=0)[0]
features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)

split_train_dur = 2
split_val_dur = 4

split_train = int(split_train_dur / 0.005)  
split_val = int(split_val_dur / 0.005)



# train_features = features_tensor[:split_train]
# val_features = features_tensor[split_train:split_val]
# test_features = features_tensor[split_val:]

# train_t = t_tensor[:split_train]
# val_t = t_tensor[split_train:split_val]
# test_t = t_tensor[split_val:]


train_data = (t_tensor[:split_train], features_tensor[:split_train])
val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
test_data = (t_tensor[split_val:], features_tensor[split_val:])

print(train_data[0].shape, val_data[0].shape, test_data[0].shape, train_data[1].shape, val_data[1].shape, test_data[1].shape)


time_points = t_tensor.numpy()  # Make sure this is a 1-D array
feature_data = features_tensor.numpy()  # Make sure this is a 2-D array

# Plotting
plt.figure(figsize=(14, 6))

# Assuming we have 4 features
plt.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)')
plt.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)')
plt.plot(time_points, feature_data[:, 2], label='Feature 3 (e_q_t)')
plt.plot(time_points, feature_data[:, 3], label='Feature 4 (e_q_st)')

plt.title('Features Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Feature Values')
plt.legend()
plt.show()




class ODEFunc(nn.Module):
    def __init__(self, N_neurons):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, 4)
        )

    def forward(self, t, y):
        return self.net(y)
    
def trainmodel(data, data2, learning_rate, num_epochs, num_neurons):
    train_losses = []
    val_losses = []
    
    t, features = data
    val_t, val_features = data2
    net = ODEFunc(N_neurons=num_neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(num_epochs):
        #training
        optimizer.zero_grad()
        pred_y = odeint(net, features[0], t)
        loss = loss_function(pred_y, features)
        loss.backward()
        optimizer.step()

        #validation
        with torch.no_grad():
            pred_y_val = odeint(net, val_features[0], val_t)
            loss_val = loss_function(pred_y_val, val_features)
        
        #storing for graphing
        train_losses.append(loss.item())
        val_losses.append(loss_val.item())


        print(f"Epoch {epoch+1}: loss = {loss.item()}")
    return net, (train_losses, val_losses)


def tictoc(func):
    def wrapper():
        t1 = time()
        func()
        t2 = time() - t1
        print(t2)
    return wrapper



network, losses = trainmodel(train_data, val_data, 0.01, 20, 50)



predicted = odeint(network, val_data[1][-1], test_data[0])
# print(predicted.shape)
MSELoss = nn.MSELoss()
evaluation_loss = MSELoss(predicted, test_data[1]).item()


print(f"Mean Squared Error Loss: {evaluation_loss}")




fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.suptitle('Actual vs Predicted Features')

feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
for i, ax in enumerate(axes.flatten()):
    ax.plot(test_data[0].numpy(), test_data[1][:, i].detach().numpy(), label='Actual ' + feature_names[i])
    ax.plot(test_data[0].numpy(), predicted[:, i].detach().numpy(), label='Predicted ' + feature_names[i], linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(feature_names[i])
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(losses[0], label='Training Loss')
plt.plot(losses[1], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()