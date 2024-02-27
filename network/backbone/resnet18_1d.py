import torch
import torch.nn as nn

from network.backbone.resnet50_1d import BottleNeckMaker, Expansion


class ResNet18_1D(nn.Module):
    def __init__(self, num_classes, gen_mode, feature_nums=41):
        """
        :param num_classes: total number of categories in the dataset
        :param feature_nums: feature dimensionality
        :param gen_mode: if or not the data generation mode, true means the network is used to generate raw dimensional data, false means it is used for classification networks or discriminators
        """
        super(ResNet18_1D, self).__init__()
        self.sep = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )


        self.layer1 = BottleNeckMaker(block_num=2, in_dim=64, hidden_dim=64, out_dim=64, if_down_sample=True)

        self.layer2 = BottleNeckMaker(block_num=2, in_dim=64, hidden_dim=128, out_dim=128, if_down_sample=True)

        self.layer3 = BottleNeckMaker(block_num=2, in_dim=128, hidden_dim=256, out_dim=256, if_down_sample=True)

        self.layer4 = BottleNeckMaker(block_num=3, in_dim=256, hidden_dim=512, out_dim=512, if_down_sample=True)

        if gen_mode:
            self.out = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, stride=3, kernel_size=3, padding=1),
                # BottleNeckMaker(block_num=3, in_dim=2048, hidden_dim=256, out_dim=512, if_down_sample=True),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                # nn.AdaptiveAvgPool1d(output_size=(1,)),
                nn.Flatten(),
                nn.Linear(512*1, feature_nums),
                Expansion(dim=1)
            )
        else:
            self.out = nn.Sequential(
                nn.AdaptiveAvgPool1d(output_size=(1,)),
                nn.Flatten(),
                nn.Linear(512, num_classes)
            )
        self.sep.apply(self.weight_init)
        self.layer1.apply(self.weight_init)
        self.layer2.apply(self.weight_init)
        self.layer3.apply(self.weight_init)
        self.layer4.apply(self.weight_init)


    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1:
            # torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.sep(x)
        # print(x.shape)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # print(x.shape)
        x = self.out(x)

        return x
