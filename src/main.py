# from pytorch_models.Torch_base_model import main as torch_base_model
from pytorch_models.TorchTest import main as torch_test_model
# from pytorch_models.Torch_Toy_Model import main as torch_toy_model


# from tensorflow_models.Tensor_base_model import main as tensor_base_model
# from tensorflow_models.TensorTest import main as tensor_test_model


#run everything from here
def main():
    torch_test_model(num_epochs=50, num_neurons=80, learning_rate=0.005, batch_range_idx=600, intermediate_pred_freq=150)
    
    # torch_base_model()

    # torch_toy_model(num_epochs=30, learning_rate=0.0003) #, intermediate_pred_freq=300)
    # torch_toy_model(num_epochs=150, learning_rate=0.01, batch_range_idx=300, mert_batch=False, intermediate_pred_freq=40)
    
    pass


if __name__ == "__main__":
    main()