
from typing import List, Tuple
from matplotlib.pyplot import figure
from torchinfo import summary
from torch.optim import Optimizer
from io import StringIO
from contextlib import redirect_stdout
from torch.types import Device

import torch
from pathlib import Path
from torch.nn import Module

from .summary import ExperimentSummary

# ============================================================================== #

class ExperimentSaver:

# ============================================================================== #

    def __init__(
        self,
        experiment_name : str,
        model_name : str,
        location : str = "experiments/", 
    )->None:
        """Intialize Instance with experiment location and name to create directory and file path

        Args:
            experiment_name (str): name of experiment
            model_name (str): name of model version
            location (str, optional): path to strore the experiment. Defaults to "experiments/".
        """
        # ----------------- Setup Path ----------------- # 
        
        self.model_name = model_name
        self.experiment_name = experiment_name
        
        self.experiment_path = Path(location) / experiment_name
        self.model_path = self.experiment_path / f"{self.model_name}.pth"
        self.summary_file = self.experiment_path / "experiment_summary.txt"
        
        # ----------------- Create Directory ----------------- # 
        
        # If the experiment name exist stop
        if self.experiment_path.is_dir():
            print(f"[ERROR] : [{self.experiment_path}] Directory exists, do nothing.")
            self.is_initialize = False
        else:
            # Create New Folder 
            print(f"[INFO] : Initialize {self.experiment_name}")
            print(f"[INFO] : Create [{self.experiment_path}] Directory")
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            self.is_initialize = True

# ============================================================================== #

    def create_experiment(
        self,
        model : Module,
        input_shape : tuple,
        dataset_size : str,
        batch_size : int,
        epochs : int,
        train_accuracy : float,
        test_accuracy : float,
        device : Device,
        optimizer : Optimizer = None,
        training_time : float = None,
        overfitting_diag : List[str] = None,
        underfitting_diag : List[str] = None,
        figures : List[figure] = None,
        extras_info : str = None
    )->None:
        """create an ExperimentSummary Object [str] who organize all experiment information

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
        
        # ----------------- Check Error ----------------- # 
         
        if not self.is_initialize:
            print(f"[ERROR] : [{self.experiment_name}] was not initialized")
            return
        
        # ----------------- Create ExperimentSummary Instance ----------------- # 
        
        summarizer = ExperimentSummary(
            model=model,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            underfitting_diag=underfitting_diag,
            overfitting_diag=overfitting_diag,
            figures=figures,
            optimizer=optimizer,
            batch_size=batch_size,
            input_shape=input_shape,
            epochs=epochs,
            device=device,
            training_time=training_time,
            dataset_size=dataset_size,
            extras_info=extras_info,
        )
        
        self.experiment_summary = summarizer.build()
        self.model = model
        self.figures = figures
        self.total_parameters = summarizer.total_parameters
        self.summarizer = summarizer
        
        # ----------------- Saving all Information ----------------- # 
        
        print(f"[INFO] : Saving {self.model.__class__.__name__} with {self.total_parameters:,} Parameters")
        print(f"[INFO] : Saving {self.model.__class__.__name__} as {self.model_name} in : [{self.model_path}]")
        torch.save(obj=self.model, f=str(self.model_path))
        
        # Saving Figures
        if self.figures:
            for index, fig in enumerate(self.figures):
                fig_path = str(self.experiment_path / f"fig_{index}")
                print(f"[INFO] : Saving Figure : [{fig_path}]")
                fig.savefig(fig_path)
        
        # Saving ExperimentSummary instance 
        with self.summary_file.open("w") as file:
            
            print(f"[INFO] : Saving Graph of Network Architecture in : [{self.summary_file}]")
            print(f"[INFO] : Saving Experiment Information in : [{self.summary_file}]") 
            file.write(self.experiment_summary)
            file.close()
        
        print(f"[INFO] : Saving {self.experiment_name} Successfully !")

# ============================================================================== #