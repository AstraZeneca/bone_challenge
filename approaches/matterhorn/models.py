import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

model_input_sizes = {
    "B0": 224,
    "B1": 240,
    "B2": 260,
    "B3": 300,
    "B4": 380,
    "B5": 456,
    "B6": 528,
    "B7": 600,
}

def adjust_first_conv_layer(model, num_input_channels):
    assert num_input_channels in [3, 6, 9, 12], "num_input_channels must be 3, 6, 9, or 12"
    
    if num_input_channels == 3:
        return  model
    print(f"Changing number of input channels from 3 to {num_input_channels}")
    original_weights = model._conv_stem.weight.data.clone()
    #print("Original weights of the first convolutional layer:")
    #print(original_weights)
    
    model._conv_stem.weight.data[:, :] = original_weights[:, :] / (num_input_channels / 3)
    # Calculate the number of times RGB weights need to be repeated
    num_repeats = num_input_channels // 3

    # Reshape the original weights to accommodate the new number of input channels while preserving the RGB pattern
    out_channels, _, kernel_size_0, kernel_size_1 = original_weights.shape
    assert isinstance(model._conv_stem, nn.Conv2d), "The original convolutional layer must be an instance of nn.Conv2d"
    model._conv_stem = nn.Conv2d(
        num_input_channels, 
        out_channels, 
        kernel_size=(kernel_size_0, kernel_size_1), 
        stride=model._conv_stem.stride, 
        padding=model._conv_stem.padding, 
        bias=model._conv_stem.bias is not None
    )
    adjusted_weights = original_weights.unsqueeze(1).repeat(1, num_repeats, 1, 1, 1).view(out_channels, num_input_channels, kernel_size_0, kernel_size_1)
    adjusted_weights /= num_repeats
    
    # Set the adjusted weights to the convolutional layer
    model._conv_stem.weight.data[:, :] = adjusted_weights    
    
    return model

class EfficientNetRegression(nn.Module):
    def __init__(self, model_name="B3", num_input_channels=3):
        super(EfficientNetRegression, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-" + model_name.lower(), num_classes=1
        )
        self.scaler = model_input_sizes[model_name]
        self.num_input_channels = num_input_channels  # Store num_input_channels
        self.activation_function = torch.tanh
        self.efficientnet = adjust_first_conv_layer(self.efficientnet, self.num_input_channels)
        
    def forward(self, x):
        x = self.efficientnet(x)
        x = self.activation_function(x)
        x = 0.5 * self.scaler * (x + 1)  # Rescale tanh to [0, scaler]
        return x

def load_model(name="B3", num_input_channels=3):
    model = EfficientNetRegression(model_name=name, num_input_channels=num_input_channels)
    return model

def load_model_and_state(model_type, num_input_channels, model_file, device):
    """
    Load a model architecture and its state from a file, and move it to the specified device.
    
    Parameters:
    - model_type (str): The type of model to load.
    - num_input_channels (int): Number of input channels for the model.
    - model_file (str): Path to the file containing the model state dictionary.
    - device (torch.device): The device to load the model onto (e.g., 'cpu' or 'cuda').
    
    Returns:
    - model (torch.nn.Module): The model with loaded state.
    """
    try:
        model = load_model(
            model_type, 
            num_input_channels=num_input_channels
        )
        model = model.to(device)        
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
