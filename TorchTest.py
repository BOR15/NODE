import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd

data = pd.read_csv("toydatafixed.csv", delimiter=';')
t_tensor = torch.tensor(data['t'].values, dtype=torch.float32)
features_tensor = torch.tensor(data.drop('t', axis=1).values, dtype=torch.float32)

#1200 is the full 6 seconds so 600 is 3 sec.
split_index = 600

train_features = features_tensor[:split_index]
predict_features = features_tensor[split_index:]
train_t = t_tensor[:split_index]
predict_t = t_tensor[split_index:]

train_data = (train_t, train_features)

class ODEFunc(nn.Module):
    def __init__(self, N_neurons):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, N_neurons),
            nn.Tanh(),
            nn.Linear(N_neurons, 4)
        )

    def forward(self, t, y):
        return self.net(y)
    
def trainmodel(data, learning_rate, num_epochs, num_neurons):
    t, features = data
    net = ODEFunc(N_neurons=num_neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_y = odeint(net, features[0], t)
        loss = loss_function(pred_y, features)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item()}")
    return net




network = trainmodel(train_data, 0.01, 10, 50)


predicted = odeint(network, train_features[-1], predict_t)
MSELoss = nn.MSELoss()
evaluation_loss = MSELoss(predicted, predict_features).item()


print(f"Mean Squared Error Loss: {evaluation_loss}")