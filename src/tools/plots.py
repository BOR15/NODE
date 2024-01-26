import matplotlib.pyplot as plt


# plotting input data
def plot_data(data_tuple, toy=False):
    time_points = data_tuple[0].numpy()  
    feature_data = data_tuple[1].numpy() 
    
    plt.figure(figsize=(14, 6))

    if toy:
        plt.plot(time_points, feature_data[:, 0], label='Feature 1 (speed)')
        plt.plot(time_points, feature_data[:, 1], label='Feature 2 (angle)')
        # plt.plot(time_points, feature_data[:, 2], label='Feature 3 (e_q_t)')
        # plt.plot(time_points, feature_data[:, 3], label='Feature 4 (e_q_st)')
    else:
        plt.plot(time_points, feature_data[:, 0], label='Angle (Delta)') 
        plt.plot(time_points, feature_data[:, 1], label='frequency (f)')
        plt.plot(time_points, feature_data[:, 2], label='Voltage (V)')
        plt.plot(time_points, feature_data[:, 3], label='Power (P)')
        plt.plot(time_points, feature_data[:, 4], label='Reactive power (Q)')

    plt.title('Features Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Values')
    plt.legend()
    

# plotting actual vs predicted values
def plot_actual_vs_predicted_full(true_y, pred_y, num_feat, min_y=None, max_y=None, lim_space=0.5, info=None, toy=False, for_torch=True):
    if for_torch:
        t = true_y[0].detach().numpy()
        true_y = true_y[1].detach().numpy()
        pred_y = pred_y.detach().numpy()
    else:
        t = true_y[0].numpy()
        true_y = true_y[1].numpy()
        pred_y = pred_y.numpy()
    

    if not min_y:
        min_y = true_y.min() - lim_space
    if not max_y:
        max_y = true_y.max() + lim_space


    total_plot_height = num_feat * 4 

    fig, axes = plt.subplots(nrows=num_feat, ncols=1, figsize=(12, total_plot_height))
    
    if not info:
        fig.suptitle('Actual vs Predicted Features Full')
    else:
        epoch, loss, *ect = info 
        fig.suptitle(f'Actual vs Predicted Features, epoch = {epoch+1}, Loss = {loss}') #TODO add the info we want in the image here.


    if toy:
        feature_names = ['speed', 'angle', 'e_q_t', 'e_q_st']
    else:
        feature_names = ['Angle (Delta)', 'frequency (f)', 'Voltage (V)', 'Power (P)', 'Reactive power (Q)']

    for i, ax in enumerate(axes):  # Update the loop to iterate over subplots
        ax.plot(t, true_y[:,i], label='Actual ' + feature_names[i])
        ax.plot(t, pred_y[:,i], label='Predicted ' + feature_names[i], linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.set_ylim(min_y, max_y)
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return plt


# plotting phase space of speed vs angle for actual vs predicted values
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
    

# plotting training vs validation loss over epochs at the end of training with optional two plots
def plot_training_vs_validation(losses, sample_freq, two_plots=True):
    #getting x axis
    x_axis = range(0, len(losses[0]) * sample_freq, sample_freq)
    
    if two_plots:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 12))

        ax1.plot(x_axis, losses[0], label='Training Loss', color='blue')  
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()

        ax2.plot(x_axis, losses[1], label='Validation Loss', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.legend()

        plt.suptitle('Training vs Validation Loss')  # Move the title above the top plot

        plt.tight_layout()
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
        x_axis = range(0, len(losses[0]) * sample_freq, sample_freq)
        
        ax1.plot(x_axis, losses[0], label='Training Loss', color='blue')  
        ax1.set_xlabel('Epoch')
        plt.title('Training vs Validation Loss')
        ax1.set_ylabel('Training Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')  

        ax2 = ax1.twinx()
        ax2.plot(x_axis, losses[1], label='Validation Loss', color='orange')

        ax2.set_ylabel('Validation Loss', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
    return plt



# def plot_training_vs_validation(losses, sample_freq, share_axis=True):
#     fig, ax1 = plt.subplots(figsize=(10, 6))
    
#     x_axis = range(0, len(losses[0]) * sample_freq, sample_freq)
    
#     ax1.plot(x_axis, losses[0], label='Training Loss', color='blue')  
#     ax1.set_xlabel('Epoch')
#     plt.title('Training vs Validation Loss')

#     if share_axis:
        
#         ax1.plot(x_axis, losses[1], label='Validation Loss', color='orange')

#         ax1.set_ylabel('Loss')
#         ax1.tick_params(axis='y')
#         plt.legend()
    
#     else:
#         ax1.set_ylabel('Training Loss', color='blue')
#         ax1.tick_params(axis='y', labelcolor='blue')
#         ax1.legend(loc='upper left')  

#         ax2 = ax1.twinx()
#         ax2.plot(x_axis, losses[1], label='Validation Loss', color='orange')

#         ax2.set_ylabel('Validation Loss', color='orange')
#         ax2.tick_params(axis='y', labelcolor='orange')
#         ax2.legend(loc='upper right')


def plot_training(loss):
    plt.plot(loss, label='Training Loss', color='blue')  
    plt.set_xlabel('Epoch')
    plt.set_ylabel('Loss')
    return plt


def plot_validation(loss):
    plt.plot(loss, label='Validation Loss', color='orange')
    plt.set_xlabel('Epoch')
    plt.set_ylabel('Loss')
    return plt

# plotting intermediate predictions during training 
def intermediate_prediction(tensor_data, predicted_intermidiate, evaluation_loss_intermidiate, num_feat, epoch):

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