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
        return x

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
        return features
    
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not last_inner.size() == feature.size():
                last_inner = self.upsample(last_inner)
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + last_inner
            results.insert(0, layer_block(last_inner))

        return results
    
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

class UpscaleNet(nn.Module):
    def __init__(self):
        super(UpscaleNet, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=7, stride=3, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=2, output_padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=6, stride=2, padding=2, output_padding=0)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x

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
        self.upsample = UpscaleNet()
        # self.upsample = nn.Upsample(scale_factor=(18, 18), mode='bilinear', align_corners=False)


    def forward(self, x):
        features = self.backbone(x)
        x = self.segmentation_head(features)
        x = torch.sigmoid(self.upsample(x))
        return x

    def unfreeze_layers(self, num_layers):
        for child in list(self.backbone.children())[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
                
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

        return out
                
def get_model(model_name, backbone_name, fpn=False, backbone_pretrained=True):
    if model_name == 'baseline':
        model = BinarySegmentationModel(fpn=fpn, backbone_name=backbone_name, backbone_pretrained=backbone_pretrained)
    if model_name == 'unet':
        model = UNet()
    return model