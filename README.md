# Framework
![](https://i.imgur.com/AmDCwzR.png)
![](https://i.imgur.com/G94ES7K.png)
## Package Usage
~~~bash
saver
├── __init__.py
├── __pycache__
├── experiment.py
├── log.py
├── model_info.py
└── summary.py

2 directories, 5 files
~~~
![](https://i.imgur.com/XZvtey1.png)
- [`ExperimentSaver()`](/saver/experiment.py)
- [`ExperimentSummary()`](/saver/summary.py)

### Training Model and Get [HistoricalTraining](https://github.com/widium/Pytorch-Training-Toolkit)
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
- output
~~~
[INFO] : Initialize experiment_2
[INFO] : Create [experiments/experiment_2] Directory
[INFO] : [experiments/log.txt] already initialized, append information inside
~~~
### Save Experiment with lot of Information
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
    last_train_accuracy=last_train_accuracy,
    last_test_accuracy=last_test_accuracy,
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
[INFO] : Saving Figure : [experiments/experiment_0/fig_0]
[INFO] : Saving Graph of Network Architecture in : [experiments/experiment_0/experiment_summary.txt]
[INFO] : Saving Experiment Information in : [experiments/experiment_0/experiment_summary.txt]
[INFO] : Saving experiment_0 Successfully !
[INFO] : Append experiment_0 information in [experiments/log.txt]
~~~
~~~bash
experiments
├── experiment_0
│   ├── experiment_summary.txt
│   └── fig_0.png
├── experiment_1
│   ├── experiment_summary.txt
│   └── fig_0.png
├── experiment_2
│   ├── experiment_summary.txt
│   └── fig_0.png
└── log.txt
~~~
- Logfile 
~~~text
****** EXPERIMENT_0 ******
- Path : [experiments/experiment_0]
- Train Accuracy : 0.81
- Test Accuracy : 0.84


****** EXPERIMENT_1 ******
- Path : [experiments/experiment_1]
- Train Accuracy : 0.89
- Test Accuracy : 0.87


****** EXPERIMENT_2 ******
- Path : [experiments/experiment_2]
- Train Accuracy : 0.96
- Test Accuracy : 0.90

~~~