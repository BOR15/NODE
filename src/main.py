# from pytorch_models.Torch_base_model import main as torch_base_model
from pytorch_models.TorchTest import main as torch_test_model
# from tensorflow_models.Tensor_base_model import main as tensor_base_model
from tensorflow_models.TensorTest import main as tensor_test_model

#run everything from here

def main():
    # torch_base_model()
    
    torch_test_model(num_epochs=30, learning_rate=0.0003)
    # tensor_test_model(num_epochs=5, learning_rate=0.0003)
    #testing git


if __name__ == "__main__":
    main()