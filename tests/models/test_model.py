import pytest
import torch
from mlops_finalproject.models.model import *

def test_output_shape():
    model = MobileNetV3Lightning(43)
    X = torch.randn(1,3,32,32)
    model.eval()
    pred = model(X)
    assert list(pred.shape) == [1,43]

def test_error_on_wrong_shape():
    model = MobileNetV3Lightning(43)
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):    
        model(torch.randn(3,32,32))
    with pytest.raises(ValueError, match='Expected 3 channels input'):
        model(torch.randn(1,1,32,32))
    with pytest.raises(ValueError, match='Expected input height between 32 and 224 pixels'):
        model(torch.randn(1,3,31,32))
    with pytest.raises(ValueError, match='Expected input height between 32 and 224 pixels'):
        model(torch.randn(1,3,225,32))
    with pytest.raises(ValueError, match='Expected input width between 32 and 224 pixels'):
        model(torch.randn(1,3,32,31))
    with pytest.raises(ValueError, match='Expected input width between 32 and 224 pixels'):
        model(torch.randn(1,3,32,225))
