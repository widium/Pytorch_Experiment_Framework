# *************************************************************************** #
#                                                                              #
#    evaluation.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/19 11:55:47 by Widium                                    #
#    Updated: 2023/04/19 11:55:47 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from typing import Tuple
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torch.types import Device
from tqdm.auto import tqdm

from .utils import logits_to_class_integer

# ============================================================================ #

def evaluation_step(
    model: Module,
    data_loader : DataLoader,
    loss_function : Module,
    metric_function : Metric,
    device : Device,
    progress_bar : bool=False

)->Tuple[float, float]:
    """Evaluate Model with Entire Dataloader

    Accumulate and Compute the Average of Loss and Metric per Batch
    
    Args:
        model (Module): Model
        data_loader (DataLoader): Dataset
        loss_function (Module): Loss Function
        metric_function (Metric): Metric Function
        device (Device): Device of Model
        progress_bar (bool, optional): Progress bar. Defaults to False.

    Returns:
        Tuple[float, float]: Average Loss and Metric per Batch
    """
    NBR_BATCH = len(data_loader)

    if progress_bar == True:
        batch_loop = tqdm(enumerate(data_loader), total=NBR_BATCH)
    else :
        batch_loop = enumerate(data_loader)

    metric_function.to(device) # Moove metric_function on device

    model.eval() # Activate the evaluation_mode

    avg_loss, avg_score = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in batch_loop:
            
            # put data on device
            X = X.to(device)
            y = y.to(device)

            y_logits = model.forward(X) # generate logits
            predicted_class = logits_to_class_integer(y_logits)
            
            # Computing Loss/Metric
            loss = loss_function(y_logits, y).item()
            score = metric_function(predicted_class, y).item()
            
            # Accumulate Loss and Score for each batch
            avg_loss += loss
            avg_score += score

            if progress_bar == True:
                batch_loop.set_description(f"Evaluating MiniBatch [{X.shape[0]}]")
        
        # compute average inside inference_mode context manager
        avg_loss /= NBR_BATCH
        avg_score /= NBR_BATCH


    return (avg_loss, avg_score)

# ============================================================================ #