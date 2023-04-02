# *************************************************************************** #
#                                                                              #
#    model_info.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/31 20:40:43 by Widium                                    #
#    Updated: 2023/03/31 20:40:43 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torchinfo import summary
from io import StringIO
from contextlib import redirect_stdout

import torch
from torch.nn import Module

from time import perf_counter
from torch.types import Device

# ================================================================================================ #

def model_summary_to_string(
    model : Module,
    batched_input_shape: tuple
)->str:
    """Catch the torchinfo summary in stdout and write it into string

    Args:
        model (Module): model for torch.summary()
        batched_input_shape (tuple): input shape of model

    Returns:
        str: output of summary() function into string
    """
    reader = StringIO()
    
    with redirect_stdout(new_target=reader):
        
        info = summary(model=model, 
                input_size=batched_input_shape,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"],
        )
        print(info)
        
    string = reader.getvalue()
    return (string)

# ================================================================================================ #

def compute_speed_of_model(
    model : Module,
    device : Device,
    batched_input_size: tuple
)->float:
    """Compute Speed of Model Inference with Dummy Data

    Args:
        model (Module): model
        device (Device): device for testing speed
        batched_input_size (tuple): input size whitout batch dimension

    Returns:
        float: prediction time in milliseconds
    """
    dummy_data = torch.randn(size=batched_input_size).to(device)
    
    model.to(device)
    
    dummy_data_device = dummy_data.device.type
    model_device = next(iter(model.parameters())).device.type
    
    # Check Device to Avoid Error
    if not (model_device == dummy_data_device):
        print(f"[INFO] : Model and Input Data are not on same Device. [Model : {model_device}], [Data : {dummy_data_device}]")
        return
        
    model.eval()
    with torch.inference_mode():
        
        start_time = perf_counter()
        model.forward(dummy_data)
        end_time = perf_counter()

    total_micro = end_time - start_time
    total_ms = total_micro * 1000

    return (total_ms)

# ================================================================================================ #

def compute_size_of_model(model : Module)->dict:
    """compute the detailed size of Pytorch Model

    Args:
        model (Module): model

    Returns:
        dict: python dictionary with 3 size 
        - `params` : accumulate size of all trainable parameters in module
        - `buffer` : accumulate size of all non-trainable tensors in module
        - `entire` : params + buffer
    """
    size = dict()
    size["params"] = 0
    size["buffer"] = 0
    
    for param in model.parameters():
        size["params"] += param.nelement() * param.element_size()

    for buffer in model.buffers():
        size["buffer"] += buffer.nelement() * buffer.element_size()

    # Convert Bytes to MegaBytes
    size["params"] /= 1024**2
    size["buffer"] /= 1024**2
    
    # compute the entire size in MegaBytes
    size["entire"] =  size["params"] + size["buffer"]
    
    return (size)

# ================================================================================================ #

def count_parameters(model : Module)->int:

    parameters_per_tensor = [param.numel() for param in model.parameters()]
    nbr_parameters = sum(parameters_per_tensor)
    return (nbr_parameters)

# ================================================================================================ #