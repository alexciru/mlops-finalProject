import pytest
from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlops_finalproject.models.predict_model
import os

####################################################################################
#                  DATA TRAIN
####################################################################################
@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/labels.pt"), reason="Data files not found")
def test_load_train_data():
    """Check whether we got all the data in train and in test"""
    # Test
    train_images = torch.load("data/processed/images.pt")
    train_labels = torch.load("data/processed/labels.pt")
    assert len(train_images) == 39209, "Train imgs size have an incorrect number of entries"
    assert len(train_labels) == 39209,  "Train labels size have an incorrect number of entries"


 
@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/labels.pt"), reason="Data files not found")
def test_load_data_shape():
    """Check whether we have the correct shape format"""
    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)  # create your datset
    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )  

    for batch, __ in trainloader:
        for img in batch:
            assert img.shape ==  torch.Size([2700]), "Img have a incorrect size"




@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/labels.pt"), reason="Data files not found")
def test_load_data_labels():
    """Check whether we loaded all the labels"""

    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)  # create your datset
    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    ) 

    unique = torch.unique(labels)
    for i in range(0, 43): assert i in unique , "data not contains all the labels"

####################################################################################
#                  DATA TEST
####################################################################################


@pytest.mark.skipif(not os.path.exists("data/raw/German/Test.csv"), reason="Data files not found")
def test_load_test_data():
    test_images, test_labels = mlops_finalproject.models.predict_model.get_data("data/raw/German")
    assert len(test_images) == len(test_labels) == 12630 , "Test size have an incorrect number of entries"
