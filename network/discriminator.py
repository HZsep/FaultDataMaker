import torch
import torch.nn as nn

from network.backbone.resnet50_1d import Expansion
from utils.net_utils import model_dict


class Discriminator(nn.Module):
    def __init__(self, num_classes, gen_mode, backbone, feature_nums):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, feature_nums)
        # 对抗结果backbone
        if backbone == 'resnet50_1d':
            adversarial_backbone = model_dict[backbone](num_classes, gen_mode, feature_nums)
            layers = list(adversarial_backbone.children())

            self.feature, self.dis_out = nn.Sequential(*layers[:-1]), nn.Sequential(
                nn.Flatten(),
                nn.Linear(6144, 1),
                # Expansion(dim=1)
            )
            self.dis_adversarial = nn.Sequential(self.feature, self.dis_out)

            # Classification results backbone
            self.dis_classifier = model_dict[backbone](num_classes, gen_mode, feature_nums)
        if backbone == 'cnn':
            self.backbone_net = model_dict[backbone](num_classes, gen_mode, feature_nums)
        self.sigmoid = nn.Sigmoid()
        self.backbone = backbone

    def forward(self, x, label):
        # embedded = self.embedding(label).unsqueeze(1)
        # x = torch.cat([x, embedded], dim=-1)
        # cnn
        # adver_out, cls_out = self.backbone(x)
        # resnet
        if self.backbone == 'resnet50_1d':
        # print(adver_out.shape)
            cls_out = self.dis_classifier(x)
            adver_out = self.dis_adversarial(x)

        else:
            # cnn
            adver_out, cls_out = self.backbone_net(x)


        return self.sigmoid(adver_out), cls_out
        # return adver_out, cls_out


# if __name__ == "__main__":
#     x = torch.randn(10, 1, 41)
#     # label = torch.tensor([random.randint(0, 19) for _ in range(10)]).unsqueeze(1)
#     # print(label)
#     dis = Discriminator(num_classes=20, gen_mode=False, backbone='resnet50_1d',feature_nums=41)
#     print(dis(x)[0].shape, dis(x)[1].shape)