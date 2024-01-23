import numpy as np
import itertools


# from pytorch_models.Torch_base_model import main as torch_base_model
from pytorch_models.TorchTest import main as torch_test_model
from pytorch_models.Torch_Gridsearch import main as torch_gridsearch_model
# from pytorch_models.Torch_Toy_Model import main as torch_toy_model


# from tensorflow_models.Tensor_base_model import main as tensor_base_model
# from tensorflow_models.TensorTest import main as tensor_test_model


#run everything from here
def main(bla=None, blaa=None, blaaa=None): #all hyperparameters get passed here as arguments
    
    # torch_test_model(num_epochs=200, num_neurons=60, learning_rate=0.01, loss_coefficient=1000, batch_range_idx=600, intermediate_pred_freq=100, mert_batch=True)
    # torch_test_model(num_epochs=20, intermediate_pred_freq=10)
    torch_gridsearch_model(num_epochs=40, epochs=[20,30])
    # torch_base_model()

    # torch_toy_model(num_epochs=30, learning_rate=0.0003) #, intermediate_pred_freq=300)
    # torch_toy_model(num_epochs=150, learning_rate=0.01, batch_range_idx=300, mert_batch=False, intermediate_pred_freq=40)
    
    return np.random.rand()


def gridsearch():
    """
    This function will do a gridsearch over the hyperparameters.
    so far it should work for all floats but gotta add something for integers. 
    
    """
    
    #number of features
    N = 3
    
    scores = np.zeros((3,) * N)


    # for initializations 
    feat1 = [1,2,4]
    feat2 = [1,2,4]
    feat3 = [1,2,4]
    
    # automatic reduction factor
    feat_red = [1/5, 1/5, 1/5]

    threshold = 0.3 #threshold for stopping tuning

    # list of autotuning features
    features = [feat1, feat2, feat3] #Do not put things in here that are options like optimizer type ect. just for floats (and its soon probably)
    tuned_features = [0] * len(features)

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
    for ii in range(10):
        # doing gridsearch
        # for i1, f1 in enumerate(feat1):
        #     for i2, f2 in enumerate(feat2):
        #         for i3, f3 in enumerate(feat3):
        #             scores[i1, i2, i3] = main(f1, f2, f3)

        # Preparing feature sets for iteration
        feature_sets = []
        for i, feature in enumerate(features):
            feature_sets.append(feature if not tuned_features[i] else [tuned_features[i]])

        print(ii, "Now trying: ", feature_sets)

        #defining score grid of right shape
        scores = np.zeros(tuple(len(f) for f in feature_sets))

        #iterating over all combinations of features
        for indices in itertools.product(*[range(len(f)) for f in feature_sets]):
            selected_features = [feature_sets[i][idx] for i, idx in enumerate(indices)]
            scores[indices] = main(*selected_features)
        
        #get best score indices
        best_indices = np.unravel_index(np.argmax(scores), scores.shape)
        
        # print(scores)
        # print(best_indices)


        # autotuning features
        for i, feature in enumerate(features):
            if not tuned_features[i]: #if feature is not tuned yet
                #if middle option is best
                if best_indices[i] == 1:
                    #checking if done tuning
                    if calculate_score_diff(i, best_indices, scores) < threshold:
                        print("DONE TUNING FEATURE", i, ":", feat_mid[i], "with score", scores[best_indices])
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

        print(ii, "Best features: ", feat_mid)
        print(ii, "Tuned features:", tuned_features)

        if all(tuned_features):
            print("All features tuned")
            break
    
    #final tune
    print("final tuning round")
    
    #full feature sets
    feature_sets = []
    for i, feature in enumerate(features):
        feature_sets.append(feature)
    
    #defining score grid of right shape
    scores = np.zeros(tuple(len(f) for f in feature_sets))

    #iterating over all combinations of features one last time
    for indices in itertools.product(*[range(len(f)) for f in feature_sets]):
        selected_features = [feature_sets[i][idx] for i, idx in enumerate(indices)]
        scores[indices] = main(*selected_features)

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
    main()