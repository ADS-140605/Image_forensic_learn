import torch
from AI.models.convnet import CameraConvNet

def test_convnet_architecture():
    # Brand classifier with 13 classes
    model = CameraConvNet(num_classes=13)
    
    # Check input
    x = torch.randn(1, 3, 128, 128)
    
    # Trace spatial sizes
    # Block 1
    x1 = model.block1(x)
    print(f"B1 Output: {x1.shape}") # Expected (1, 96, 30, 30)
    assert x1.shape == (1, 96, 30, 30)
    
    # Block 2
    x2 = model.block2(x1)
    print(f"B2 Output: {x2.shape}") # Expected (1, 64, 13, 13)
    assert x2.shape == (1, 64, 13, 13)
    
    # Block 3
    x3 = model.block3(x2)
    print(f"B3 Output: {x3.shape}") # Expected (1, 64, 4, 4)
    assert x3.shape == (1, 64, 4, 4)
    
    # Block 4
    x4 = model.block4(x3)
    print(f"B4 Output: {x4.shape}") # Expected (1, 128, 4, 4)
    assert x4.shape == (1, 128, 4, 4)
    
    # Flatten
    xf = x4.flatten(1)
    print(f"Flattened size: {xf.shape[1]}")
    assert xf.shape[1] == 2048
    
    # Final output
    out = model.classifier(xf)
    print(f"Output size: {out.shape[1]}")
    assert out.shape[1] == 13
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params}")
    # Paper said 2,585,149 for 13 classes.
    assert params == 2585149

if __name__ == "__main__":
    test_convnet_architecture()
    print("Architecture verified successfully!")
