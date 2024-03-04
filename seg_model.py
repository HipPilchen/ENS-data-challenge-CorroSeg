import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from dataloader import CorroSeg
from tqdm import tqdm
import torch

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 1)
        
    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = SegModel()
        
    corro_seg = CorroSeg('data', 'y_train.csv', shuffle = True,
                 batch_size = 64, valid_ratio = 0.1, transform_img=None,  
                 transform_test=None, test_params={'batch_size': 1, 'shuffle': False})
    train_loader, val_loader, test_loader = corro_seg.get_loaders()
    
    for i, (x, y, z) in tqdm(enumerate(train_loader)):
        pred = model(x)
        print(pred.shape)
