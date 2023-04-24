# *************************************************************************** #
#                                                                              #
#    train.py                                                                  #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/19 11:55:49 by Widium                                    #
#    Updated: 2023/04/19 11:55:49 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from typing import Dict, Tuple
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torch.types import Device
from torch.optim import Optimizer
from tqdm.auto import tqdm
from time import perf_counter

from .evaluation import evaluation_step
from .history import HistoricalTraining
from .utils import logits_to_class_integer

# ============================================================================ #

def training_step(
    model: Module, 
    data_loader : DataLoader,
    loss_function : Module,
    optimizer : Optimizer,
    metric_function : Metric,
    device : Device,
    progress_bar : bool=False,

)->Tuple[float, float]:
    """Training Step for entire DataLoader
    
    Accumulate and Compute the Average of Loss and Metric per Batch
    
    Args:
        `model` (Module): Model
        `data_loader` (DataLoader): Dataset 
        `loss_function` (Module): Loss Function
        `optimizer` (Optimizer): Optimizer
        `metric_function` (Metric): Metric Function
        `device` (Device): Device of Model
        `progress_bar` (bool, optional): Progress bar. Defaults to False.

    Returns:
        Tuple[float, float]: Return Average Loss and Metric per Batch
    """
    NBR_BATCH = len(data_loader)

    if progress_bar == True:
        batch_loop = tqdm(enumerate(data_loader), total=NBR_BATCH)
    else :
        batch_loop = enumerate(data_loader)

    # Initialize avg value
    avg_loss, avg_score = 0, 0

    # Change Device of Metric Function
    metric_function.to(device)


    # Activate Training Mode
    model.train()

    for batch, (X, y) in batch_loop:

        # Change device of data in batch
        X = X.to(device)
        y = y.to(device)

        y_logits = model.forward(X) # Generate Logits

        # Compute Loss/Metric
        loss = loss_function(y_logits, y)
        
        predicted_classes = logits_to_class_integer(y_logits)
        score = metric_function(predicted_classes, y).item()
        
        # Accumulate Loss/Metric per Batch
        avg_score += score
        avg_loss += loss.item()
        
        optimizer.zero_grad() # Reset Gradient

        loss.backward() # Backpropagate the loss

        optimizer.step() # Update Parameters
        
        if progress_bar == True:
            batch_loop.set_description(f"Training MiniBatch [{X.shape[0]}]")


    # Compute Avg Value with Nbr of Batch
    avg_loss /= NBR_BATCH
    avg_score /= NBR_BATCH

    return (avg_loss, avg_score)

# ============================================================================ #

def train(
    model : Module,
    train_dataloader : DataLoader,
    test_dataloader : DataLoader,
    optimizer : Optimizer,
    loss_function : Module,
    metric_function : Metric,
    device : Device,
    epochs : int = 1,

)->Dict:
    """Train and Evaluate Model with Train and Test Dataloader 

    Track model performance with HistoricalTraining instance
    Store every epoch result in HistoricalTraining instance
    display information of each epoch
    at the end display curves of Metric and Loss
    
    Args:
        `model` (Module): model
        `train_dataloader` (DataLoader): Train Dataset
        `test_dataloader` (DataLoader): Test Dataset
        `optimizer` (Optimizer): Optimizer
        `loss_function` (Module): Loss Function
        `metric_function` (Metric): Metric Function
        `device` (Device): device of model
        `epochs` (int, optional): max epoch to train. Defaults to 1.
        `name` (str, optional): name of model. Defaults to "model_0".

    Returns:
        Dict[HistoricalTraining, str, str, str]: Performance and name of Model
    """
    history = HistoricalTraining(max_epochs=epochs)

    # Initialize List for each tracked Value
    history["Train Loss"] = list()
    history["Train Accuracy"] = list()
    history["Val Loss"] = list()
    history["Val Accuracy"] = list()

    start_time = perf_counter()

    epoch_loop = tqdm(range(epochs))

    for epoch in epoch_loop:

        train_loss, train_score = training_step(
            model=model,
            data_loader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            metric_function=metric_function,
            device=device,
            progress_bar=False
        )

        val_loss, val_score = evaluation_step(
            model=model,
            data_loader=test_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            device=device,
            progress_bar=False
        )
        
        history["Train Loss"].append(train_loss)
        history["Train Accuracy"].append(train_score)
        history["Val Loss"].append(val_loss)
        history["Val Accuracy"].append(val_score)
        
        history.display_info(epoch)

    end_time = perf_counter()
    
    history["Training Time"] = end_time - start_time
    history.diagnostic()
    history.plot_curves()

    return (history)

# ============================================================================ #