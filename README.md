Project Description 
==============================


## Overview
[Project pipeline.](reports/figures/graph2.png)

## Overall goal of the project  

We want to familiarize ourselves with industry standards machine learning software delivery pipeline by deploying a simple Traffic Sign recognition model. To do so we expect to rely on the tools reviewed in the course to carry out the different stages of a project deployment: Design, Development, and Operations. Keeping the focus on organization, reproducibility, efficient version control, debugging, and cloud integrations.   

## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)

PyTorch Image Model (TIMM)

## How do you intend to include the framework into your project   

The model selected can use with trained or without weights. The idea is to test the weighted version to test the performance and maybe use the feature extraction features of the framework to try to improve the accuracy.

## What data are you going to run on (initially, may change)  

[German Traffic Sign Recognition Benchmark GTSRB.](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)  
The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class image classification benchmark in the domain of advanced driver assistance systems and autonomous driving. It consists of 50000 images distributed among 43 classes.

What deep learning models do you expect to use? We expect to use MobileNetV3, which is a convolutional neural network (CNN) architecture primarily designed for efficient and accurate image recognition on mobile devices. It uses depthwise separable convolutions, inverted residuals, linear bottlenecks, and mobile blocks to reduce the computational complexity of the network and improve efficiency.

# Make dataset
Optional `-n` argument to process only specified amount of images

`python mlops_finalproject/data/make_dataset.py  data/raw/German/Train  data/processed/ -n 300`


# Predict in inference

It needs
- Argument for the model checkpoint
- Argument with a directory with only images

`python mlops_finalproject/models/predict_model.py models/trained_model_timm_lightning.pt data/raw/German/Test/`

# Docker
## Trainer
Build Docker with

`docker build -f trainer.dockerfile . -t trainer_project:latest`

Execute with

`docker run -e WANDB_API_KEY=<wandb_api_key> trainer_project:latest`


# Inference Pytorchserve
1. Build docker

`docker build -f inference.dockerfile --tag=europe-west1-docker.pkg.dev/mlops-finalproject/inference-mnet/serve-mnet .`

2. Execute docker

` docker run -p 8080:8080 --name=local_mnet_inference99 europe-west1-docker.pkg.dev/mlops-finalproject/inference-mnet/serve-mnist`

3. Send to get prediction (copy that to terminal)

```
cat > instances.json <<END
{
  "instances": [
    {
      "data": {
        "b64": "b64": "$(base64 --wrap=0 data/raw/German/Test/00000.png)"
      }
    }
  ]
}
END

curl -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @instances.json \
  localhost:8080/predictions/mnist
```

### Predict
Not yet implemented

------------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so mlops_finalproject can be imported
    ├── mlops_finalproject                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes mlops_finalproject a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │ │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages (using conda)
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use wandb to log training progress and other important metrics/artifacts in your code
* [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2
* [x] Write unit tests related to the data part of your code ' Alex
* [ ] Write unit tests related to model construction  - Michael
* [ ] Calculate the coverage.    - easy command all
* [ ] Get some continuous integration running on the github repository - Henning  
CLOUD -- MEET OTHER DAY ??
* [ ] (optional) Create a new project on `gcp` and invite all group members to it
* [ ] Create a data storage on `gcp` for you data
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training on `gcp`
* [ ] Play around with distributed data loading
* [ ] (optional) Play around with distributed model training
* [ ] Play around with quantization and compilation for you trained models

### Week 3

* [ ] Deployed your model locally using TorchServe
* [ ] Checked how robust your model is towards data drifting
* [ ] Deployed your model using `gcp`
* [ ] Monitored the system of your deployed model
* [ ] Monitored the performance of your deployed model

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Create a presentation explaining your project
* [ ] Uploaded all your code to github
* [ ] (extra) Implemented pre*commit hooks for your project repository
* [ ] (extra) Used Optuna to run hyperparameter optimization on your model



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
