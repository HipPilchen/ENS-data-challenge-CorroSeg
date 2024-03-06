import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        x = self.model(x)['out']
        return torch.sigmoid(x)
    

if __name__ == "__main__":
    # model = SegModel()
    base_model = models.resnet50(pretrained=False)
    
    for name, layer in base_model.named_children():
        print(name)
        print(layer)
        
    
    
    # print(model)
    # x = torch.randn(1, 3, 36, 36)
    # print(model(x).shape)
        
    