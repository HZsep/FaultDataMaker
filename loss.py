import torch.nn as nn
import torch
from utils.net_utils import compute_gradient_penalty
import torch.nn.functional as F


class Dis_loss(nn.Module):
    def __init__(self):
        super(Dis_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, real_data, label, randn_input, dis, gen, lambda_gp):
        # fake_data = gen(randn_input, label)
        # fake_validity, fake_cls = dis(fake_data, label)
        # real_validity, real_cls = dis(real_data, label)
        # gradient_penalty = compute_gradient_penalty(dis, real_data.data, fake_data.data, label)
        # dis_gan_loss = torch.mean(fake_validity) - torch.mean(real_validity)  # ls
        # cls_loss_real = self.criterion(real_cls, label)
        # gan_loss = dis_gan_loss + lambda_gp * gradient_penalty


        # ACGAN
        real_labels = torch.ones(label.size(0), 1).cuda()
        fake_labels = torch.zeros(label.size(0), 1).cuda()
        real_outputs, real_cls = dis(real_data, label)
        d_loss_real = self.bce_loss(real_outputs, real_labels)
        fake_images = gen(randn_input, label)
        fake_outputs, fake_cls = dis(fake_images, label)
        d_loss_fake = self.bce_loss(fake_outputs, fake_labels)
        cls_loss_real = self.criterion(real_cls, label)
        gan_loss = d_loss_real + d_loss_fake

        total_loss = gan_loss + cls_loss_real


        return total_loss, {
            "total_loss": total_loss.item(),
            "gan_loss": gan_loss.item(),
            "cls_loss_real": cls_loss_real.item(),
            # "gradient_penalty": gradient_penalty.item()
        }


class Gen_loss(nn.Module):
    def __init__(self):
        super(Gen_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, real_data, label, randn_input, dis, gen):
        # ACGAN
        # fake_data = gen(randn_input, label)
        # fake_validity, fake_cls = dis(fake_data, label)
        # cls_loss_fake = self.criterion(fake_cls, label)
        # gen_gan_loss = -torch.mean(fake_validity)
        # Ac-gan
        real_labels = torch.ones(label.size(0), 1).cuda()
        fake_data = gen(randn_input, label)
        fake_validity, fake_cls = dis(fake_data, label)
        gen_gan_loss = self.bce_loss(fake_validity, real_labels)
        cls_loss_fake = self.criterion(fake_cls, label)

        true_item_norm = torch.nn.functional.normalize(input=real_data, p=2, dim=-1)
        fake_item_norm = torch.nn.functional.normalize(input=fake_data, p=2, dim=-1)
        norm_loss = self.mse(true_item_norm, fake_item_norm)

        gen_total_loss = gen_gan_loss + cls_loss_fake + norm_loss * 1e4
        return gen_total_loss, {
            "total_loss": gen_total_loss.item(),
            "gan_loss": gen_gan_loss.item(),
            "cls_loss_fake": cls_loss_fake.item(),
            'norm_loss': norm_loss.item()
        }
