import numpy as np
import math
import itertools
import pandas as pd
from tools.logsystem import getnewrunid

# from pytorch_models.Torch_base_model import main as torch_base_model
from pytorch_models.TorchTest import main as torch_test_model
from pytorch_models.Torch_Gridsearch import main as torch_gridsearch_model
# from pytorch_models.Torch_Toy_Model import main as torch_toy_model


# from tensorflow_models.Tensor_base_model import main as tensor_base_model
# from tensorflow_models.TensorTest import main as tensor_test_model

def main():
    # torch_test_model(num_epochs=200, num_neurons=60, learning_rate=0.01, loss_coef=1000, batch_range_idx=600, intermediate_pred_freq=100, mert_batch=True)
    torch_test_model(num_epochs=20, intermediate_pred_freq=10)
    # torch_base_model()

    # torch_toy_model(num_epochs=30, learning_rate=0.0003) #, intermediate_pred_freq=300)
    # torch_toy_model(num_epochs=150, learning_rate=0.01, batch_range_idx=300, mert_batch=False, intermediate_pred_freq=40)
    pass



#run everything from here
def gridmain(learning_rate, num_neurons, batch_size, batch_dur_idx, batch_range_idx, lmbda, loss_coefficient, rel_tol,
             abs_tol, val_freq, regu, ODEmethod, epochs=None): #all hyperparameters get passed here as arguments
    if not epochs:
        print("Epochs not received properly")
        return None
    else:
        num_epochs = epochs[-1]
        epochs = epochs[:-1]
    score = []

    runid = getnewrunid()

    dataset1 = "clean_mean0_data_g1.pt"
    dataset2 = "clean_mean0_data_g2.pt"
    dataset3 = "clean_mean0_data_g8.pt"
    datasets = [dataset1, dataset2, dataset3]

    for dataset in datasets:
        score.append(torch_gridsearch_model(dataset, runid, num_epochs=num_epochs, epochs=epochs, learning_rate=learning_rate,
                                            num_neurons=num_neurons, batch_size=batch_size, batch_dur_idx=batch_dur_idx,
                                            batch_range_idx=batch_range_idx, lmbda=lmbda, loss_coefficient=loss_coefficient,
                                            rel_tol=rel_tol, abs_tol=abs_tol, val_freq=val_freq, regu=regu, ODEmethod=ODEmethod))


    # score.append(torch_gridsearch_model(1, runid, num_epochs=num_epochs, epochs=epochs, learning_rate = learning_rate, num_neurons = num_neurons,
    #                                     batch_size=batch_size, batch_dur_idx=batch_dur_idx, batch_range_idx=batch_range_idx, lmbda=lmbda,
    #                                     loss_coefficient=loss_coefficient, rel_tol=rel_tol, abs_tol=abs_tol, val_freq=val_freq, regu=regu,
    #                                     ODEmethod=ODEmethod))
    # score.append(torch_gridsearch_model(2, runid, num_epochs=num_epochs, epochs=epochs, learning_rate = learning_rate, num_neurons = num_neurons,
    #                                     batch_size=batch_size, batch_dur_idx=batch_dur_idx, batch_range_idx=batch_range_idx, lmbda=lmbda,
    #                                     loss_coefficient=loss_coefficient, rel_tol=rel_tol, abs_tol=abs_tol, val_freq=val_freq, regu=regu,
    #                                     ODEmethod=ODEmethod))
    # score.append(torch_gridsearch_model(3, runid, num_epochs=num_epochs, epochs=epochs, learning_rate = learning_rate, num_neurons = num_neurons,
    #                                     batch_size=batch_size, batch_dur_idx=batch_dur_idx, batch_range_idx=batch_range_idx, lmbda=lmbda,
    #                                     loss_coefficient=loss_coefficient, rel_tol=rel_tol, abs_tol=abs_tol, val_freq=val_freq, regu=regu,
    #                                     ODEmethod=ODEmethod))

    score = np.mean(np.array(score), axis=0)
    
    frechet, time = score

    frechet_coeff = 1
    time_coeff = 1
    frechet_pwr = 2
    time_pwr = 2
    frechet = 1 / (1 + (frechet_coeff * frechet)**frechet_pwr) #inverse fretchet distance so that higher is better and between 0 and 1
    time = 1 / (1 + (time_coeff * time)**time_pwr) #inverse time so that higher is better and between 0 and 1

    final_score = frechet * time  #kind of like an and
    #or this?
    # final_score = accuracy + loss + time  #kind of like an or

    # return np.random.rand()
    return final_score


def gridsearch():
    """
    This function will do a gridsearch over the hyperparameters.
    so far it should work for all floats but gotta add something for integers. 
    
    """
    
    #number of iterations for automatic tuning
    iterations = 1
    
    #epochs
    epochs = [10, 20]

    feature_names = ['learning_rate']

    # initial values autotuning features
    learning_rate = [0.001, 0.0001, 0.00001]



    # list of autotuning features
    features = [learning_rate] #Do not put things in here that are options like optimizer type ect. just for floats (and its soon probably)
    is_int = [0, 1]

    #initial values non autotuning features
    num_neurons = [50]
    batch_size = [40, 50, 60]
    batch_dur_idx = [10, 20]
    batch_range_idx = [400, 500, 600]
    lmbda = [5e-3]
    loss_coefficient = [1]
    rel_tol = [1e-7]
    abs_tol = [1e-9]
    val_freq = [5]
    regu = [None]


    ODEmethod = ['dopri5']

    non_auto = [num_neurons, batch_size, batch_dur_idx, batch_range_idx, lmbda, loss_coefficient, rel_tol, abs_tol, val_freq, regu, ODEmethod]

    #list of all features
    all_features = [*features, *non_auto]
    # is_int.extend([0] * (len(all_features) - len(features)))

    # automatic reduction factor
    feat_red = [1/5, 1/5, 1/5]

    threshold = 0.3 #threshold for stopping tuning

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
            scores[indices] = gridmain(*selected_features, epochs=epochs)  ##THIS COMMENT IS HERE BECAUSE I KEEP SCROLLLING PAST THIS LINE 
        
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
        scores[indices] = gridmain(*selected_features, epochs=epochs) ##THIS COMMENT IS HERE BECAUSE I KEEP SCROLLLING PAST THIS LINE 

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
    gridsearch()
    # gridmain()
    #main()