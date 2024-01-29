import matplotlib.pyplot as plt
import torch

data = '/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/stretched_downsampled_197_mean0_data_g2.pt'
unstretched_time = "/Users/laetitiaguerin/Library/CloudStorage/OneDrive-Personal/Documents/BSc Nanobiology/Year 4/Capstone Project/Github repository/NODE/downsampled_197_timepoints_data_g2.pt"

data_tuple = torch.load(data)
unstretched_time_tuple = torch.load(unstretched_time)


def plot_no_predictions(true_y, num_feat, figtitle, min_y=None, max_y=None, lim_space=0.5,
                        for_torch=True, unstretched=None):
    if for_torch:
        t = true_y[0].detach().numpy()
        true_y = true_y[1].detach().numpy()
    else:
        t = true_y[0].numpy()
        true_y = true_y[1].numpy()

    if not min_y:
        min_y = true_y.min() - lim_space
    if not max_y:
        max_y = true_y.max() + lim_space

    total_plot_height = num_feat * 4 

    fig, axes = plt.subplots(nrows=num_feat, ncols=1, figsize=(12, total_plot_height))
    
    plt.suptitle(figtitle, fontsize=16, fontweight='bold')

    feature_names = ['Angle (Delta)', 'frequency (f)', 'Voltage (V)', 'Power (P)', 'Reactive power (Q)']

    for i, ax in enumerate(axes):  # Update the loop to iterate over subplots
        ax.plot(t, true_y[:,i], label= feature_names[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(feature_names[i])
        ax.set_ylim(min_y, max_y)
        ax.legend()
    plt.tight_layout(rect=[0.01, 0.03, 1, 0.95])
    plt.show()


plot_no_predictions(data_tuple, num_feat=5, figtitle='Stretched time system dynamics')
plot_no_predictions(unstretched_time_tuple, num_feat=5, figtitle='Unstretched time system dynamics')