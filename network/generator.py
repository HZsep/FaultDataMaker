import torch
import torch.nn as nn
from utils.net_utils import model_dict
from network.backbone.resnet50_1d import Expansion
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class Generator(nn.Module):
    def __init__(self, num_classes, gen_mode, backbone, feature_nums):
        super(Generator, self).__init__()
        self.backbone = model_dict[backbone](num_classes, gen_mode, feature_nums)
        self.sep, self.layer1, self.layer2, self.layer3, self.layer4 = \
            self.backbone.sep, self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_classes, embedding_dim=feature_nums),
            Expansion(dim=1)
        )

        self.dilated_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.attention1 = AttentionModule(1024)

        self.dilated_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.attention2 = AttentionModule(512)

        self.up_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.up_conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, inputs, label):
        label_embed = self.embedding(label)
        x = inputs * label_embed
        x = self.backbone(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dilated_conv1(x)
        x = self.attention1(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dilated_conv2(x)
        x = self.attention2(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up_conv1(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up_conv2(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up_conv3(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up_conv4(x)
        # print(x.shape)
        x = F.interpolate(x, size =41)
        # print(x.shape)
        return x

if __name__ == "__main__":
    a = torch.randn(2, 1, 41)
    label = torch.tensor([19, 8])
    gen = Generator(num_classes=20, gen_mode=True, feature_nums=41, backbone='resnet18_1d')
    res = gen(a, label)
