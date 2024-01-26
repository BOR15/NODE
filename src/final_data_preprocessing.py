from tools.data_processing import main as data_preprocessing_main

starts = [150]              # The point after the first peak that you cut to, if this is known
shifts = [0]                # should be between 0 and 4, where it indicates 1 out the 5 generator spots 
suffixes = ["Final_test"]   # This helps to name the file in your repository, should be a list
data_path = "Input_Data\Raw_Data\Dynamics107h16.csv"

def preprocessing():
    """This function is the main function for preprocessing the raw data. (How the data should be is in READ ME)"""
    data_preprocessing_main(
        data_path=data_path,                        
        shifts=shifts,                             
        starts=starts,                              
        suffixes=suffixes, 
        path_root = '',
        normalization = [None],
        stretching = False, 
        downsampling = None, 
        interpolation = 'linear',
        num_samples_interpolation = None
                            )
if __name__ == "__main__":
    preprocessing()