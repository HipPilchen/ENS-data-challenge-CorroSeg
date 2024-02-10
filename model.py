import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class BinarySegmentationModel(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(BinarySegmentationModel, self).__init__()
        # Use ResNet as an example backbone
        self.backbone = models.resnet18(pretrained=backbone_pretrained)
        
        # Omit the last classification layer
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))
        
        # Add segmentation-specific layers
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Adjust the number of channels based on the backbone's output
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)  # 1-channel output for binary segmentation
        
        # Upsampling layer to match the input image resolution
        self.upsample = nn.Upsample(scale_factor=(18, 18), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Feature extraction through the backbone
        x = self.backbone(x)
        
        # Applying segmentation layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Use sigmoid for binary classification
        
        # Upsampling to get the segmentation map of the same size as the input image
        x = self.upsample(x)
        return x
    
    def unfreeze_layers(self, num_layers):
        # Get total number of layers in the backbone
        total_layers = len(list(self.backbone.children()))
        
        # Ensure num_layers does not exceed total number of layers
        num_layers = min(num_layers, total_layers)
        
        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last `num_layers` layers
        for child in list(self.backbone.children())[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
                
                
def get_model(model_name, backbone_pretrained=True):
    if model_name == 'resnet18':
        model = BinarySegmentationModel(backbone_pretrained=backbone_pretrained)
    return model