import copy
import torch.nn as nn
from network.backbone.resnet18_1d import ResNet18_1D
from network.backbone.resnet50_1d import ResNet50_1D
from network.backbone.resnet101_1d import ResNet101_1D
from network.backbone.convnet import CNN
import torch
import numpy as np

model_dict = {
    'resnet18_1d': ResNet18_1D,
    'resnet50_1d': ResNet50_1D,
    'resnet101_1d': ResNet101_1D,
    'cnn': CNN
}


def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


# def compute_gradient_penalty(dis, real_data, fake_data):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     alpha = torch.randn(size=(real_data.size(0), 1, 1), requires_grad=True)
#     interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
#     d_interpolates = dis(interpolates)
#     fake = torch.ones(size=(real_data.size(0), 1), requires_grad=False)
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

def compute_gradient_penalty(dis, real_data, fake_data,label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_data.shape[0], 1, 1)).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    out, cls = dis(interpolates, label)  # dis(interpolates).requires_grad_(False)
    # d_interpolates.requires_grad_(False)
    fake = torch.full((real_data.shape[0], 1), 1.0, dtype=real_data.dtype, requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=fake)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
