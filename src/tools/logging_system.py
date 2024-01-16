import csv
import os
import pandas as pd

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

    def log_model(self, hyperparameters, plot_urls=None):
        """Log the hyperparameters and maybe future URL's?"""
        if self.logging_enabled:
            with (open(self.log_file, mode='a', newline='') as file):
                writer = csv.DictWriter(file, fieldnames=['Hyperparameters', 'Plot URLs'])
                row ={'Hyperparameters': hyperparameters, 'Plot URLs': ','.join(plot_urls) if plot_urls else ''}

                writer.writerow(row)




logger = ModelLogger(logging_enabled=True) #enable or disable the logger for your run
hyperparameters = {"learning rate": 0.001, 'batch_size': 50}
plot_urls = ['http://example.com/plot1', 'http://example.com/plot2']

logger.log_model(hyperparameters, plot_urls)

# print(logger)