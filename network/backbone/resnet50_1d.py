import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, if_down_sample):
        super(BottleNeck, self).__init__()

        self.if_down_sample = if_down_sample
        down_stride = 2 if if_down_sample else 1

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=down_stride, padding=1
        )

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        if if_down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=down_stride),
                nn.BatchNorm1d(out_dim)
            )




    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.if_down_sample:
            residual = self.shortcut(residual)
        x = self.relu(torch.add(residual, x))
        return x


class Expansion(nn.Module):
    def __init__(self, dim=1):
        super(Expansion, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(dim=self.dim)


class BottleNeckMaker(nn.Module):
    def __init__(self, block_num, in_dim, hidden_dim, out_dim, if_down_sample):
        super(BottleNeckMaker, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.add_module(
            name="Bottleneck1",
            module=BottleNeck(in_dim, hidden_dim, out_dim, if_down_sample)
        )
        for i in range(1, block_num):
            module_name = "Bottleneck" + str(i + 1)
            self.layer.add_module(
                name=module_name,
                module=BottleNeck(out_dim, hidden_dim, out_dim, False)
            )
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class ResNet50_1D(nn.Module):
    def __init__(self, num_classes, gen_mode, feature_nums=41):
        """
        :param num_classes: total number of categories in the dataset
        :param feature_nums: feature dimensionality
        :param gen_mode: if or not the data generation mode, true means the network is used to generate raw dimensional data, false means it is used for classification networks or discriminators
        """
        super(ResNet50_1D, self).__init__()
        self.sep = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),

            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm1d(64),
            # nn.ReLU()
        )


        self.layer1 = BottleNeckMaker(block_num=3, in_dim=64, hidden_dim=64, out_dim=256, if_down_sample=True)

        self.layer2 = BottleNeckMaker(block_num=4, in_dim=256, hidden_dim=128, out_dim=512, if_down_sample=True)

        self.layer3 = BottleNeckMaker(block_num=23, in_dim=512, hidden_dim=256, out_dim=1024, if_down_sample=True)

        self.layer4 = BottleNeckMaker(block_num=3, in_dim=1024, hidden_dim=512, out_dim=2048, if_down_sample=True)

        if gen_mode:
            self.out = nn.Sequential(
                nn.Conv1d(in_channels=2048, out_channels=512, stride=3, kernel_size=3, padding=1),
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
                nn.Linear(2048, num_classes)
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

        # 分类
        # print(x.shape)
        x = self.out(x)

        return x


# self.layer1 = nn.Sequential(
        #     BottleNeck(in_dim=64, hidden_dim=64, out_dim=256, if_down_sample=if_down_sample, down_stride=down_stride),
        #     BottleNeck(in_dim=256, hidden_dim=64, out_dim=256, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=256, hidden_dim=64, out_dim=256, if_down_sample=False, down_stride=None)
        # )
        #
        # self.layer2 = nn.Sequential(
        #     BottleNeck(in_dim=256, hidden_dim=128, out_dim=512, if_down_sample=if_down_sample, down_stride=down_stride),
        #     BottleNeck(in_dim=512, hidden_dim=128, out_dim=512, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=512, hidden_dim=128, out_dim=512, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=512, hidden_dim=128, out_dim=512, if_down_sample=False, down_stride=None)
        # )
        #
        # self.layer3 = nn.Sequential(
        #     BottleNeck(in_dim=512, hidden_dim=256, out_dim=1024, if_down_sample=if_down_sample,
        #                down_stride=down_stride),
        #     BottleNeck(in_dim=1024, hidden_dim=256, out_dim=1024, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=1024, hidden_dim=256, out_dim=1024, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=1024, hidden_dim=256, out_dim=1024, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=1024, hidden_dim=256, out_dim=1024, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=1024, hidden_dim=256, out_dim=1024, if_down_sample=False, down_stride=None)
        # )
        #
        # self.layer4 = nn.Sequential(
        #     BottleNeck(in_dim=1024, hidden_dim=512, out_dim=2048, if_down_sample=if_down_sample,
        #                down_stride=down_stride),
        #     BottleNeck(in_dim=2048, hidden_dim=512, out_dim=2048, if_down_sample=False, down_stride=None),
        #     BottleNeck(in_dim=2048, hidden_dim=512, out_dim=2048, if_down_sample=False, down_stride=None)
        # )

# if __name__ == "__main__":
#     a = torch.randn(10, 1, 41)
#     net = ResNet50_1D(num_classes=20, gen_mode=True, feature_nums=41)
#     print(net(a).shape)
