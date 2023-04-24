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
    experiment_name="experiment_0",
    model_name="efficient_net_b0",
    location="experiments/"
)
~~~
### Create and Save Experiment with lot of Information
method detail here : [`ExperimentSaver.create_experiment()`](/saver/experiment.py)
~~~python
num_epochs = len(history["Epochs"])
last_train_accuracy = history["Train Accuracy"][-1]
last_test_accuracy = history["Val Accuracy"][-1]
underfitting_diagnostic = history["Bias and UnderFitting"]
overfitting_diagnostic = history["Bias and UnderFitting"]
training_time = history["Training Time"]
experiment_figures = [history["Curve Figure"]]

# ****************************************************

saver.create_experiment(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset_size="10%",
    batch_size=BATCH_SIZE,
    epochs=num_epochs,
    train_accuracy=last_train_accuracy,
    test_accuracy=last_test_accuracy,
    underfitting_diag=underfitting_diagnostic,
    overfitting_diag=overfitting_diagnostic,
    figures=experiment_figures,
    optimizer=optimizer,
    device=device,
    training_time=training_time,
    extras_info=""
)
~~~
### Output
~~~bash
[INFO] : Initialize experiment_0
[INFO] : Create [experiments/experiment_0] Directory
[INFO] : Saving EfficientNet with 4,011,391 Parameters
[INFO] : Saving EfficientNet as efficient_net_b0 in : [experiments/experiment_0/efficient_net_b0.pth]
[INFO] : Saving Figure : [experiments/experiment_0/fig_0]
[INFO] : Saving Graph of Network Architecture in : [experiments/experiment_0/experiment_summary.txt]
[INFO] : Saving Experiment Information in : [experiments/experiment_0/experiment_summary.txt]
[INFO] : Saving experiment_0 Successfully !
[INFO] : Append experiment_0 information in [experiments/log.txt]
~~~
~~~bash
experiments
├── experiment_0
│   ├── efficient_net_b0.pth
│   ├── experiment_summary.txt
│   └── fig_0.png
├── experiment_1
│   ├── efficient_net_b0.pth
│   ├── experiment_summary.txt
│   └── fig_0.png
└── log.txt

3 directories, 7 files
~~~