import torch
import torch.nn as nn

from network.backbone.resnet50_1d import BottleNeckMaker, Expansion


class CNN(nn.Module):
    def __init__(self, num_classes, gen_mode, feature_nums=100):
        """
        :param num_classes: total number of categories in the dataset
        :param feature_nums: feature dimensionality
        :param gen_mode: if or not the data generation mode, true means the network is used to generate raw dimensional data, false means it is used for classification networks or discriminators
        """
        super(CNN, self).__init__()
        base_channels = 128 * 8
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=base_channels, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm1d(base_channels*2),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm1d(base_channels*4),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=base_channels*4, out_channels=base_channels*2, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm1d(base_channels*8),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=3, padding=1, stride=3),
            # nn.BatchNorm1d(base_channels*16),
            nn.ReLU()
        )

        if gen_mode:
            self.out = nn.Sequential(
                nn.Conv1d(in_channels=base_channels * 16, out_channels=base_channels, kernel_size=3, padding=1,
                          stride=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.BatchNorm1d(base_channels),
                nn.Linear(base_channels * 2, feature_nums),
                Expansion(dim=1)
            )
            self.out.apply(self.weight_init)
        else:
            self.adver_out = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels*2, 1)
            )

            self.cls_out = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels*2, num_classes)
            )


        self.layer1.apply(self.weight_init)
        self.layer2.apply(self.weight_init)
        self.layer3.apply(self.weight_init)
        self.layer4.apply(self.weight_init)
        self.layer5.apply(self.weight_init)

        self.gen_mode = gen_mode


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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # 分类
        if self.gen_mode:
            x = self.out(x)
            return x
        else:
            out1 = self.adver_out(x)
            # print("out1.shape", out1.shape)
            out2 = self.cls_out(x)
            # print("out2.shape", out2.shape)
            return out1, out2
        # print(x.shape)



# if __name__ == "__main__":
#     a = torch.randn(10, 1, 41)
#     net = CNN(num_classes=20, gen_mode=True, feature_nums=41)
#     print(net(a).shape)
