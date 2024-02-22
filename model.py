import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        # x = self.adaptive_pool(x)
        return torch.sigmoid(x)

class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone, fpn=False, selected_layers=None):
        super(EfficientNetBackbone, self).__init__()
        self.features = backbone.features
        self.selected_layers = selected_layers if selected_layers else [2, 4, 6, 8]

    def forward(self, x):
        # features = []
        # for i, layer in enumerate(self.features):
        #     x = layer(x)
        #     if i in self.selected_layers:
        #         features.append(x)
        features = self.features(x)
        return torch.sigmoid(features)
    
def get_backbone(backbone_name, fpn=False, pretrained=True):
    backbone_factory = {
            'resnet18': lambda: models.resnet18(pretrained=pretrained),
            'resnet34': lambda: models.resnet34(pretrained=pretrained),
            'resnet50': lambda: models.resnet50(pretrained=pretrained),
            'resnet101': lambda: models.resnet101(pretrained=pretrained),
            'efficientnet-b0': lambda: models.efficientnet_b0(pretrained=pretrained),
            'efficientnet-v2-s': lambda: models.efficientnet_v2_s(pretrained=pretrained),
            'efficientnet-v2-m': lambda: models.efficientnet_v2_m(pretrained=pretrained),
            'efficientnet-v2-l': lambda: models.efficientnet_v2_l(pretrained=pretrained),
        }

    if backbone_name in backbone_factory:
        backbone = backbone_factory[backbone_name]()
        if 'resnet' in backbone_name:
            backbone = ResNetBackbone(backbone)
        elif 'efficientnet' in backbone_name:
            backbone = EfficientNetBackbone(backbone, fpn=fpn)   
        else:
            raise ValueError("Unsupported backbone")
        # Dynamically determine the output channels
        dummy_input = torch.rand(1, 3, 36, 36)  # Assuming a typical input size for ResNet
        output_features = backbone(dummy_input)
        if isinstance(output_features, torch.Tensor):
            output_channels = output_features.size(1)
        else:  # List of tensors, adjust according to your segmentation head design
            output_channels = sum([feature.size(1) for feature in output_features])
    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported")
    
    return backbone, output_channels

class BinarySegmentationModel(nn.Module):
    def __init__(self, fpn=False, backbone_name='resnet50', backbone_pretrained=True):
        super(BinarySegmentationModel, self).__init__()
        
        self.backbone, output_channels = get_backbone(backbone_name, fpn=fpn, pretrained=backbone_pretrained)
 
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(output_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.upsample = nn.Upsample(scale_factor=(18, 18), mode='bilinear', align_corners=False)


    def forward(self, x):
        features = self.backbone(x)
        x = self.segmentation_head(features)
        x = torch.sigmoid(self.upsample(x))
        return x

    def unfreeze_layers(self, num_layers):
        for child in list(self.backbone.children())[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
                
                


# Définir le modèle CNN
class baseline_CNN(nn.Module):
    def __init__(self):
        super(baseline_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding = (1,1))  # Couche 1 : Convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = (1,1))  # Couche 2 : Convolution
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = (1,1))  # Couche 3 : Convolution
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding = (1,1))  # Couche 4 : Convolution
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding = (1,1))  # Couche 5 : Convolution
        self.fc1 = nn.Linear(512 * 36 * 36, 36*36)  # Couche Dense
        self.fc2 = nn.Linear(36*36, 36*36)  # Couche de sortie

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        # print(x.shape)
        x = x.view(-1, 512 * 36 * 36)  # Aplatir les données
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).view(-1,1,36,36)  # Sigmoid pour la classification binaire
        return x


                
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Load a pretrained ResNet and use it as the encoder
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.encoder1 = nn.Sequential(*self.base_layers[:3])  # Initial conv + bn + relu + maxpool
        self.encoder2 = nn.Sequential(*self.base_layers[3:5])  # Layer 1
        self.encoder3 = self.base_layers[5]  # Layer 2
        self.encoder4 = self.base_layers[6]  # Layer 3
        self.encoder5 = self.base_layers[7]  # Layer 4

        # Decoder layers
        self.decoder4 = self.conv_block(2048, 1024)
        self.decoder3 = self.conv_block(1024, 512)
        self.decoder2 = self.conv_block(512, 256)
        self.decoder1 = self.conv_block(256, 64)

        # Final classifier
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)  # Adjusted
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Decoder with skip connections
        dec4 = self.decoder4(enc5)
        dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = dec4 + enc4  # Now add after resizing

        dec3 = self.decoder3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = dec3 + enc3  # Resize then add

        dec2 = self.decoder2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = dec2 + enc2  # Resize then add

        dec1 = self.decoder1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = dec1 + enc1  # Resize then add

        # Final classification layer
        out = self.final_conv(dec1)
        # Upsample back to the input size
        out = F.interpolate(out, size=(36, 36), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class SegmentationCNN(nn.Module):
    def __init__(self):
        super(SegmentationCNN, self).__init__()
        self.conv1 = ConvBlock(3, 64)  # Assuming input images are RGB
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        # Upsampling + Convolution layers to restore the original image size with 2 channels for segmentation
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = ConvBlock(512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = ConvBlock(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = ConvBlock(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = ConvBlock(64, 64)

        # Final 1x1 convolution to get 2 channels for the two labels
        self.final_conv = nn.Conv2d(64, 1, 1, stride = 16)

    def forward(self, x):
        # Pass input through the CNN blocks
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Upsample and pass through additional convolutions
        x = self.upconv4(x5)
        x = self.conv6(x + F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=False))  # Skip connection

        x = self.upconv3(x)
        x = self.conv7(x + F.interpolate(x3, size=x.size()[2:], mode='bilinear', align_corners=False))  # Skip connection

        x = self.upconv2(x)
        x = self.conv8(x + F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False))  # Skip connection

        x = self.upconv1(x)
        x = self.conv9(x + F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=False))  # Skip connection

        # Final convolution to get 2 channels
        out = self.final_conv(x)

        return torch.sigmoid(out)
    
                
def get_model(model_name, backbone_name, fpn=False, backbone_pretrained=True):
    if model_name == 'first_model':
        model = BinarySegmentationModel(fpn=fpn, backbone_name=backbone_name, backbone_pretrained=backbone_pretrained)
    elif model_name == 'unet':
        model = UNet()
    elif model_name == 'cnn':
        model = SegmentationCNN()
    elif model_name == 'baseline_cnn':
        model = baseline_CNN()
    return model