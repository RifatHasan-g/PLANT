import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

class LiteRASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_classes=3):  # Change num_classes to 3
        super(LiteRASPP, self).__init__()
        self.scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, x):
        x = self.scale(x)
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.maxpool(x))
        avg_out = self.fc(self.avgpool(x))
        out = max_out + avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels_deep, in_channels_shallow, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.cbam = CBAM(in_channels_shallow)
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels_deep + in_channels_shallow, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, deep_feature, shallow_feature):
        deep_feature = self.up(deep_feature)
        shallow_feature = self.cbam(shallow_feature)
        return self.concat_conv(torch.cat([deep_feature, shallow_feature], dim=1))

class ModifiedDeepLabV3(nn.Module):
    def __init__(self, backbone, classifier, decoder, num_classes):
        super(ModifiedDeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.decoder = decoder
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        result = {'out': x}
        return result

def prepare_model(num_classes=3):  # Change num_classes to 3
    backbone_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    backbone = backbone_model.features
    last_channel = backbone[-1].out_channels
    classifier = LiteRASPP(last_channel, 256, num_classes)
    decoder = FeatureFusionModule(last_channel, 256, 256)
    model = ModifiedDeepLabV3(backbone, classifier, decoder, num_classes)
    return model
