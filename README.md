# Framework
![](https://i.imgur.com/AmDCwzR.png)
![](https://i.imgur.com/G94ES7K.png)
## Package Usage
~~~bash
saver
├── experiment.py
├── __init__.py
├── model_info.py
├── __pycache__
└── summary.py

2 directories, 4 files
~~~
![](https://i.imgur.com/6Q576Dj.png)
- [`ExperimentSaver()`](/saver/experiment.py)
- [`ExperimentSummary()`](/saver/summary.py)

### Training Model and Get [HistoricalTraining](https://github.com/widium/Historical_Training)
~~~python
from training.train import train
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score
from torch.optim import Adam

BATCH_SIZE = 32
INPUT_SHAPE = (COLOR, 224, 224)

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = CrossEntropyLoss()
metric_function = F1Score(task="multiclass", num_classes=NBR_CLASS)

history = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_function=loss_function,
    metric_function=metric_function,
    device=device,
    epochs=5
)
~~~
### Initialize Experiment with location and name
~~~python
from saver.experiment import ExperimentSaver

saver = ExperimentSaver(
    experiment_name="experiment_1",
    model_name="model_1",
    location="experiments/"
)
~~~
### Create and Save Experiment with lot of Information
~~~python
saver.create_experiment(
    model=model,
    input_shape=INPUT_SHAPE,
    dataset_size="10%",
    batch_size=BATCH_SIZE,
    epochs=len(history["Epochs"]),
    train_accuracy=history["Train Accuracy"][-1],
    test_accuracy=history["Val Accuracy"][-1],
    underfitting_diag=history.diagnostic_results["Bias and UnderFitting"],
    overfitting_diag=history.diagnostic_results["Variance and OverFitting"],
    figures=[history.curve_figure],
    optimizer=optimizer,
    device=device,
    training_time=history["Training Time"],
    extras_info="Add Regularization on last Layer give me Better Accuracy on validation set"
)
~~~
method detail here : [`ExperimentSaver.create_experiment()`](/saver/experiment.py)
~~~python
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
~~~
### Output
~~~bash
[INFO] : Initialize experiment_1
[INFO] : Create [experiments/experiment_1] Directory
[INFO] : Saving EfficientNetFoodClassifier with 4,011,391 Parameters
[INFO] : Saving EfficientNetFoodClassifier as model_1 in : [experiments/experiment_1/model_1.pth]
[INFO] : Saving Figure : [experiments/experiment_1/fig_0]
[INFO] : Saving Graph of Network Architecture in : [experiments/experiment_1/experiment_summary.txt]
[INFO] : Saving Experiment Information in : [experiments/experiment_1/experiment_summary.txt]
[INFO] : Saving experiment_1 Successfully !
~~~
~~~bash
experiments
├── experiment_0
│   ├── experiment_summary.txt
│   ├── fig_0.png
│   └── model_0.pth
└── experiment_1
    ├── experiment_summary.txt
    ├── fig_0.png
    └── model_1.pth

3 directories, 6 files
~~~