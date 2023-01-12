Project Description 
==============================




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
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
