import csv
import os
import pandas

class ModelLogger:
    def __init__(self, log_file='model_log.csv', logging_enabled=True):
        self.log_file = log_file
        self.logging_enabled = logging_enabled
        self.create_log_file()

    def create_log_file(self):
        """Create the log file """
        if self.logging_enabled and not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Hyperparameters", "Plot URLs"])

    def log_model(self, hyperparameters, plot_urls):
        """Log the hyperparameters and maybe future URL's?"""
        if self.logging_enabled:
            with open(self.log_file, mode='a', newline='') as file:
                # writer = csv.writer(file)
                df_original = pd.read_csv(file)
                combined_df = df_original.append(hyperparameters, ignore_index=True)
                combined_df.to_csv(self.log_file)
                # writer.writerow([str(hyperparameters), ', '.join(plot_urls)])
                #also add the loss metric
                # writer.writerow([str(metric())])




logger = ModelLogger(logging_enabled=True) #enable or disable the logger for your run


