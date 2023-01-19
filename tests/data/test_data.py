import pytest
from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlops_finalproject.models.predict_model
import os

####################################################################################
#                  DATA TRAIN
####################################################################################
@pytest.mark.skipif(not os.path.exists("data/processed/train/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train/labels.pt"), reason="Data files not found")
def test_load_train_data():
    """Check whether we got all the data in train and in test"""
    # Test
    train_images = torch.load("data/processed/train/images.pt")
    train_labels = torch.load("data/processed/train/labels.pt")
    assert len(train_images) == 39209, "Train imgs size have an incorrect number of entries"
    assert len(train_labels) == 39209,  "Train labels size have an incorrect number of entries"


 
@pytest.mark.skipif(not os.path.exists("data/processed/train/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train/labels.pt"), reason="Data files not found")
def test_load_data_shape():
    """Check whether we have the correct shape format"""
    images = torch.load("data/processed/train/images.pt")
    labels = torch.load("data/processed/train/labels.pt")

    train_dataset = TensorDataset(images, labels)  # create your datset
    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )  

    for batch, __ in trainloader:
        for img in batch:
            assert img.shape ==  torch.Size([3, 32, 32]), "Img have a incorrect size"




@pytest.mark.skipif(not os.path.exists("data/processed/train/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train/labels.pt"), reason="Data files not found")
def test_load_data_labels():
    """Check whether we loaded all the labels"""

    images = torch.load("data/processed/train/images.pt")
    labels = torch.load("data/processed/train/labels.pt")

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
    
    test_images = torch.load("data/processed/test/images.pt")
    test_labels = torch.load("data/processed/test/labels.pt")

    assert len(test_images) == len(test_labels) == 12630 , "Test size have an incorrect number of entries"
