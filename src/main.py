# from pytorch_models.Torch_base_model import main as torch_base_model
# from pytorch_models.TorchTest import main as torch_test_model

# from tensorflow_models.Tensor_base_model import main as tensor_base_model
from tensorflow_models.TensorTest import main as tensor_test_model
from tensorflow_models.Torch_Toy_Model import main as torch_toy_test_model
#run everything from here

def main():=
    # torch_base_model()
    
    torch_toy_test_model(num_epochs=30, learning_rate=0.0003) #, intermediate_pred_freq=300)
    tensor_test_model(num_epochs=30, learning_rate=0.0003)
    #testing git


if __name__ == "__main__":
    main()