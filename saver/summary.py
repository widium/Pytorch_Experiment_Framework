# *************************************************************************** #
#                                                                              #
#    summary.py                                                                #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/31 20:41:05 by Widium                                    #
#    Updated: 2023/03/31 20:41:05 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from typing import List, Tuple
from torch.types import Device
from torch.optim import Optimizer
from matplotlib.pyplot import figure

from pathlib import Path
from torch.nn import Module

from .model_info import model_summary_to_string
from .model_info import compute_speed_of_model
from .model_info import compute_size_of_model
from .model_info import count_parameters

# ============================================================================== #

class ExperimentSummary:
    """Create a String with all information of model and experiment"""

# ============================================================================== #

    def __init__(
        self,
        model : Module,
        input_shape : tuple,
        dataset_size : str,
        batch_size : int,
        epochs : int,
        last_train_accuracy : float,
        last_test_accuracy : float,
        device : Device,
        optimizer : Optimizer = None,
        training_time : float = None,
        overfitting_diag : List[str] = None,
        underfitting_diag : List[str] = None,
        figures : List[figure] = None,
        extras_info : str = None
    )->None:
        """Intialize Instance with all information that define the model like :
           - training performance
           - dataset parameters
           - shapes
           - ect...

        Args:
            `model` (Module): model
            `input_shape` (tuple): input shape unbatched of model
            `dataset_size` (str): string define size of dataset in percentage compared to initial dataset
            `batch_size` (int): batch size with which the model was trained
            `epochs` (int): nbr epochs
            `train_accuracy` (float): last training accuracy
            `test_accuracy` (float): last test accuracy
            `device` (Device): device with which the model was trained
            `optimizer` (Optimizer, optional): optimizer used for update parameters. Defaults to None.
            `training_time` (float, optional): time in second for training loop. Defaults to None.
            `overfitting_diag` (List[str], optional): overfitting diagnostic of training process. Defaults to None.
            `underfitting_diag` (List[str], optional): underfitting diagnostic of training process. Defaults to None.
            `figures` (List[figure], optional): list of matplotlib figure. Defaults to None.
            `extras_info` (str, optional): Additonal notes to append to summary. Defaults to None.
        """
        self.model = model
        self.device = device
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.last_train_accuracy = last_train_accuracy
        self.last_test_accuracy = last_test_accuracy
        self.optimizer = optimizer
        self.training_time = training_time
        self.overfitting_diag = overfitting_diag
        self.underfitting_diag = underfitting_diag
        self.figures = figures     
        self.extras_info = extras_info  
        
        # ----------------- Create Additional Model Information ----------------- #  
        
        self.model_network = model_summary_to_string(
            model=self.model,
            batched_input_shape=input_shape
        )
        
        self.model_speed = compute_speed_of_model(
            model=self.model,
            device=device,
            batched_input_size=input_shape
        )
        
        self.model_size = compute_size_of_model(model)
        self.total_parameters = count_parameters(model)
        self.str_training_time = f"{training_time:.3f} second" if training_time else None

# ============================================================================== #  
    
    def build(self)->Tuple[str]:
        """Create a multiple fstring with information of experimentation Concatenated in Global string

        Returns:
            Tuple[str]: gloab string with multiple fstring join separated by '\\n'
        """
        diagnostic_results = f"\n***** DIAGNOSTIC *****\n"
        diagnostic_results += f"- Bias and UnderFitting : {', '.join(self.underfitting_diag) if self.underfitting_diag else None}\n"
        diagnostic_results += f"- Variance and OverFitting : {', '.join(self.overfitting_diag) if self.overfitting_diag else None}\n"

        learning_parameters = f"\n***** LEARNING PARAMETERS *****\n"
        learning_parameters += f"- Training Dataset Size : {self.dataset_size}\n"
        learning_parameters += f"- Batch Size : {self.batch_size}\n"
        learning_parameters += f"- Number of Epochs : {self.epochs}\n"
        learning_parameters += f"- Optimizer Parameters : {self.optimizer.defaults if self.optimizer else None}\n"

        metrics_performance = f"\n***** METRICS PERFORMANCE *****\n"
        metrics_performance += f"- Last Train Accuracy : {self.last_train_accuracy:.3f}\n"
        metrics_performance += f"- Last Test Accuracy : {self.last_test_accuracy:.3f}\n"
        
        speed_performance = f"\n***** SPEED PERFORMANCE *****\n"
        speed_performance += f"- Device in Training : {self.device}\n"
        speed_performance += f"- Training Time : {self.str_training_time}\n"
        speed_performance += f"- Prediction Time : {self.model_speed:.3f} ms\n"

        size = f"\n***** MODEL SIZE *****\n"
        size += f"- Total Parameters : {self.total_parameters:,}\n"
        size += f"- Model Parameters size: {self.model_size['params']:.3f} (MB)\n"
        size += f"- Model Utils size: {self.model_size['buffer']:.3f} (MB)\n"
        size += f"- Model Entire Size: {self.model_size['entire']:.3f} (MB)\n"

        notes = f"\n***** NOTES *****\n"
        notes += f"- {self.extras_info}\n"
        
        network = f"\n***** MODEL NETWORK ARCHITECTURE *****\n"
        network += self.model_network
        
        # ----------------- Store all F-string into global string ----------------- #
        
        self.experiment_summary = "\n".join(
            [
                metrics_performance,
                diagnostic_results,
                learning_parameters,
                speed_performance, 
                size,
                notes,
                network
            ]
        )
        
        return (self.experiment_summary)

# ============================================================================== #