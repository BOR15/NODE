from tools.data_processing import main as data_preprocessing_main

starts = []     # The point after the first peak that you cut to, if this is known
shifts = []     # should be between 0 and 4, where it indicates 1 out the 5 generator spots 
suffixes = []   # This helps to name the file in your repository
data_path = []

def preprocessing():
    """This function is the main function for preprocessing the raw data. (How the data should be is in READ ME)"""
    data_preprocessing_main(data_path=data_path, shifts=shifts, starts=starts, suffixes=suffixes, path_root=..., interpolation_type=..., num_samples_interpolation=[200])


if __name__ == "__main__":
    preprocessing()