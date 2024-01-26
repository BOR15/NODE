from tools.data_processing import main as data_preprocessing_main

# These lists should align per index (first element of each list should be the information for one signal, and so on)
starts = []     # List of indices at which the signal is cut from (start is removed because of bad measurements--find manually on graph)
shifts = []     # List of integers between 0 and 4, where it indicates 1 out the 5 meaurement sources--find in CSV file; first source after time axis has shift 0 
suffixes = []   # List of strings to add as suffix to file names (one suffix per source; types of processing are already named in the data_preprocessing function)
data_path = ''  # Relative path to the data you want to process

def preprocessing():
    '''
    Inputs (including the starts, shifts, suffixes, and data_path described above):
        path_root: string; if relative path does not work, input full path to source folder (here, NODE)
        normalization: list of strings; options are 'mean0std1' and 'normalize'
        stretching: boolean; linspace the time axis of your dataset
        downsampling: integer or None; if stretching is on, input integer here to downsample to this input (give or take some because of math)
        interpolation: if you choose to interpolate, what kind of interpolation?
        num_samples_interpolation: integer or None; if integer is given, interpolate to that number of samples with given interpolation type
    '''
    data_preprocessing_main(
        data_path,  # if relative path does not work, input full path here
        shifts, 
        starts, 
        suffixes, 
        path_root= "", 
        normalization=[None], 
        stretching=False,  
        downsampling=None, 
        interpolation='linear',  
        num_samples_interpolation=None 
        )


if __name__ == "__main__":
    preprocessing()