import numpy as np
import math
import itertools
import pandas as pd
from tools.logsystem import getnewrunid

from pytorch_models.Torch_base_model import main as torch_base_model
# from pytorch_models.TorchTest import main as torch_test_model
from pytorch_models.Torch_Gridsearch import main as torch_gridsearch_model
from pytorch_models.final_model import main as final_model
# from pytorch_models.Torch_Toy_Model import main as torch_toy_model


# from tensorflow_models.Tensor_base_model import main as tensor_base_model
# from tensorflow_models.TensorTest import main as tensor_test_model

data_path = 'mean0_interpolated_Final_train_copy_2_200_samples.pt'

def main():
    
    # torch_base_model(num_neurons=50, num_epochs=50, learning_rate=0.01, train_duration=1.5, val_shift=0.1)
    runid = getnewrunid()
    final_model(dataset= data_path,      # This should be the name of the file that was created by "data_preprocessing_main(...)"
                runid=runid, 
                num_neurons=50,                         # amount of neurons in layers.
                num_epochs=800,                         # total number of epochs
                epochs=[100, 200, 300, 400,500,600,700],        # intermediate results
                learning_rate=0.003, 
                loss_coefficient=10,                     # makes the loss artificially bigger
                batch_size=50,                          # How many samples per batch
                batch_dur_idx=8,                        # index indicates how many seconds of the data we use per batch
                batch_range_idx=60,                     # index the amount of data for sampling training batches
                rel_tol=1e-7,
                abs_tol=1e-9, 
                val_freq=5, 
                lmbda=5e-3,                             # regularization factor
                ODEmethod="dopri5", 
                interpolation_type="quadratic", 
                num_samples_interpolation=400, 
                regu= None,  
                mert_batch_scuffed=False, 
                mert_batch=False,
                intermediate_pred_freq=0, 
                live_intermediate_pred=False, 
                live_plot=False, 
                savemodel=False, 
                savepredict=False
                )



############################################################
####                                                    ####
#### BELOW IS FOR GRIDSEARCHING, NOT NEEDED TO RUN MAIN ####
####                                                    ####
############################################################
    
#run everything from here
def gridmain(learning_rate, num_neurons, batch_size, batch_dur_idx, batch_range_idx, lmbda, loss_coefficient, rel_tol,
             abs_tol, val_freq, regu, ODEmethod, normalization, interpolation_density, epochs=None): #all hyperparameters get passed here as arguments
    if not epochs:
        print("Epochs not received properly")
        return None
    else:
        num_epochs = epochs[-1]
        epochs = epochs[:-1]
    score = []

    runid = getnewrunid()

    if normalization == "mean0std1":
        if interpolation_density == None:
            dataset1 = "clean_mean0_data_g1.pt"
            dataset2 = "clean_mean0_data_g2.pt"
            dataset3 = "clean_mean0_data_g8.pt"
        elif interpolation_density == 100:
            dataset1 = "mean0_interpolated_g1_100_samples.pt"
            dataset2 = "mean0_interpolated_g2_100_samples.pt"
            dataset3 = "mean0_interpolated_g8_100_samples.pt"
        elif interpolation_density == 400:
            dataset1 = "mean0_interpolated_g1_400_samples.pt"
            dataset2 = "mean0_interpolated_g2_400_samples.pt"
            dataset3 = "mean0_interpolated_g8_400_samples.pt"
        elif interpolation_density == 1200:
            dataset1 = "mean0_interpolated_g1_1200_samples.pt"
            dataset2 = "mean0_interpolated_g2_1200_samples.pt"
            dataset3 = "mean0_interpolated_g8_1200_samples.pt"
        elif interpolation_density == "stretch":
            dataset1 = "stretched_mean0_data_g1.pt"
            dataset2 = "stretched_mean0_data_g2.pt"
            dataset3 = "stretched_mean0_data_g8.pt"
        else:
            print("Interpolation density not recognized")
            return None
        
    elif normalization == "norm0_1":
        if interpolation_density == None:
            dataset1 = "clean_normalized_data_g1.pt"
            dataset2 = "clean_normalized_data_g2.pt"
            dataset3 = "clean_normalized_data_g8.pt"
        elif interpolation_density == 100:
            dataset1 = "normalized_interpolated_g1_100_samples.pt"
            dataset2 = "normalized_interpolated_g2_100_samples.pt"
            dataset3 = "normalized_interpolated_g8_100_samples.pt"
        elif interpolation_density == 200:
            dataset1 = "normalized_interpolated_g1_200_samples.pt"
            dataset2 = "normalized_interpolated_g2_200_samples.pt"
            dataset3 = "normalized_interpolated_g8_200_samples.pt"
        elif interpolation_density == 400:
            dataset1 = "normalized_interpolated_g1_400_samples.pt"
            dataset2 = "normalized_interpolated_g2_400_samples.pt"
            dataset3 = "normalized_interpolated_g8_400_samples.pt"
        elif interpolation_density == 1200:
            dataset1 = "normalized_interpolated_g1_1200_samples.pt"
            dataset2 = "normalized_interpolated_g2_1200_samples.pt"
            dataset3 = "normalized_interpolated_g8_1200_samples.pt"
        elif interpolation_density == "stretch":
            dataset1 = "stretched_normalized_data_g1.pt"
            dataset2 = "stretched_normalized_data_g2.pt"
            dataset3 = "stretched_normalized_data_g8.pt"
        else:
            print("Interpolation density not recognized")
            return None
        
    else:
        print("Normalization not recognized")
        return None

    if interpolation_density == None:
        batch_dur_idx /= 18/800
        batch_range_idx /= 18/800
    elif interpolation_density == 100:
        batch_dur_idx /= 18/100
        batch_range_idx /= 18/100
    elif interpolation_density == 200:
        batch_dur_idx /= 18/200
        batch_range_idx /= 18/200
    elif interpolation_density == 400:
        batch_dur_idx /= 18/400
        batch_range_idx /= 18/400
    elif interpolation_density == 1200:
        batch_dur_idx /= 18/1200
        batch_range_idx /= 18/1200
    elif interpolation_density == "stretch":
        batch_dur_idx /= 18/800
        batch_range_idx /= 18/800
    
    
    batch_dur_idx = int(batch_dur_idx)
    batch_range_idx = int(batch_range_idx)

    if batch_dur_idx < 3:
        batch_dur_idx = 3
    if batch_range_idx < batch_size:
        batch_range_idx = batch_size

    print("batch_dur_idx: ", batch_dur_idx, "batch_range_idx: ", batch_range_idx)
    
    datasets = [dataset1, dataset2, dataset3]

    for dataset in datasets:
        score.append(torch_gridsearch_model(dataset, runid, num_epochs=num_epochs, epochs=epochs, learning_rate=learning_rate,
                                            num_neurons=num_neurons, batch_size=batch_size, batch_dur_idx=batch_dur_idx,
                                            batch_range_idx=batch_range_idx, lmbda=lmbda, loss_coefficient=loss_coefficient,
                                            rel_tol=rel_tol, abs_tol=abs_tol, val_freq=val_freq, regu=regu, ODEmethod=ODEmethod))


    score = np.mean(np.array(score), axis=0)
    
    frechet, time = score

    if normalization == "mean0std1":
        frechet_coeff = 5
    elif normalization == "norm0_1":
        frechet_coeff = 15

    time_coeff = 1
    frechet_pwr = 2.5
    time_pwr = 2.5
    frechet = 1 / (1 + (frechet_coeff * frechet)**frechet_pwr) #inverse fretchet distance so that higher is better and between 0 and 1
    time = 1 / (1 + (time_coeff * time)**time_pwr) #inverse time so that higher is better and between 0 and 1

    final_score = frechet * time  
    
    return final_score


def gridsearch():
    """
    This function will do a gridsearch over the hyperparameters.
    so far it should work for all floats but gotta add something for integers. 
    
    """
    
    #number of iterations for automatic tuning
    iterations = 1
    
    #epochs
    epochs =  [10,20,50,100,150,200]

    feature_names = []

    # initial values autotuning features
    



    # list of autotuning features
    features = [] #Do not put things in here that are options like optimizer type ect. 
    is_int = [0]

    #initial values non autotuning features
    learning_rate = [0.003]     #[0.1, 0.001, 0.00001]
    num_neurons = [50]          #[10, 25, 50]
    batch_size =  [60]          # base = 10 #[5, 10, 25, 50] 
    batch_dur_idx = [0.35]      # chosen base = 0.5 #[0.1, 0.3, 0.5]
    batch_range_idx = [4]       #[2,5,10]
    lmbda = [5e-3]
    loss_coefficient = [100] #[1, 10]
    rel_tol = [1e-7]
    abs_tol = [1e-9]
    val_freq = [5]
    regu = [None]
    
    #Dataset things
    normalization = ["mean0std1"] #["mean0std1", "norm0_1"]
    interpolation_density = [100] #[None, 100, 400, "stretch"]


    ODEmethod = ['dopri5']

    non_auto = [learning_rate, num_neurons, batch_size, batch_dur_idx, batch_range_idx, lmbda, loss_coefficient, rel_tol, abs_tol, val_freq, regu, ODEmethod, normalization, interpolation_density]

    #list of all features
    all_features = [*features, *non_auto]
    # is_int.extend([0] * (len(all_features) - len(features)))

    # automatic reduction factor
    feat_red = [1/10, 1/10, 1/10]

    threshold = 0 #threshold for stopping tuning

    tuned_features = [0] * len(all_features)

    feat_diff = []
    feat_diff1 = []
    feat_diff2 = []
    feat_mid = []
    
    #getting diff and mid
    for feature in features:
        feat_mid.append(feature[1])
        
        feat_diff2.append(feature[2]-feature[1])
        feat_diff1.append(feature[1]-feature[0])

        
        # feat_diff.append((feature[2] - feature[0]) / 2)

    
    #gridsearch loop
    for ii in range(iterations):
        # Preparing feature sets for iteration
        feature_sets = []
        for i, feature in enumerate(all_features):
            print(tuned_features[i])  # should be 0???
            feature_sets.append(feature if not tuned_features[i] else [tuned_features[i]])

        #printing feature sets and total number of combinations
        print(ii, "Now trying: ", feature_sets)
        print("Number of hyperparameter sets", math.prod([len(feature) for feature in feature_sets]))

        #defining score grid of right shape
        scores = np.zeros(tuple(len(f) for f in feature_sets))

        #iterating over all combinations of features
        for indices in itertools.product(*[range(len(f)) for f in feature_sets]):
            selected_features = [feature_sets[i][idx] for i, idx in enumerate(indices)]
            print("features: ", feature_names)
            print("values:   ", selected_features)
            scores[indices] = gridmain(*selected_features, epochs=epochs)  
        
        #get best score indices
        best_indices = np.unravel_index(np.argmax(scores), scores.shape)
        
        # print(scores)
        # print(best_indices)

        #printing best features
        best_features = [feature_sets[i][idx] for i, idx in enumerate(best_indices)]
        print(ii, "Best features: ", best_features)

        
        # autotuning features
        for i, feature in enumerate(features):
            if not tuned_features[i]: #if feature is not tuned yet
                #if middle option is best
                if best_indices[i] == 1:
                    #checking if done tuning
                    if calculate_score_diff(i, best_indices, scores) < threshold:
                        print("DONE TUNING FEATURE", i, ": Value;", feat_mid[i], "with score", scores[best_indices])
                        #if done saving best value
                        tuned_features[i] = feat_mid[i]
                        continue #continue to next feature                        
                    else: #if not done tuning: tunne more!
                        #narrowing range of feature if middle was best
                        feat_diff1[i] *= feat_red[i]
                        feat_diff2[i] *= feat_red[i]
                else:#if middle option is not best: redefine middle
                    feat_mid[i] = feature[best_indices[i]]
                #defining new range
                features[i] = [feat_mid[i] - feat_diff1[i], feat_mid[i], feat_mid[i] + feat_diff2[i]]
                if is_int[i]:
                    features[i] = [round(x) for x in features[i]]
        
        #updating all_features to match autotuned features
        all_features = [*features, *non_auto]

        print(ii, "Tuned features:", tuned_features)

        if all(tuned_features):
            print("All features tuned")
            break
    if iterations > 1:
        #final tune
        print("final tuning round")
        
        #full feature sets
        feature_sets = []
        for i, feature in enumerate(all_features):
            feature_sets.append(feature)
        
        #defining score grid of right shape
        scores = np.zeros(tuple(len(f) for f in feature_sets))

        #iterating over all combinations of features one last time
        for indices in itertools.product(*[range(len(f)) for f in feature_sets]):
            selected_features = [feature_sets[i][idx] for i, idx in enumerate(indices)]
            scores[indices] = gridmain(*selected_features, epochs=epochs) 

        best_indices = np.unravel_index(np.argmax(scores), scores.shape)
        final_features = [feature_sets[i][idx] for i, idx in enumerate(best_indices)]
        print("Final features: ", final_features)

        
        
def calculate_score_diff(feature_idx, best_indices, scores):
    temp_indices = list(best_indices)
    
    # sc
    temp_indices[feature_idx] += 1
    score_high = scores[tuple(temp_indices)]

    temp_indices[feature_idx] -= 2
    score_low = scores[tuple(temp_indices)]

    score_diff = abs(score_high - score_low)
    return score_diff


if __name__ == "__main__":
    # gridsearch()
    # gridmain()
    main()