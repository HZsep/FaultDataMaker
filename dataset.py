import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import data_creator
from utils.utils import getConfig, save2json
import numpy as np


class CommonDataset(Dataset):
    def __init__(self, flag):
        super(CommonDataset, self).__init__()
        assert flag in ['train', 'test', 'valid'], "'flag' must be set to one of ['train', 'test', 'valid']"
        train, valid = data_creator()
        data_dict = {'train': train, 'valid': valid}
        self.data = data_dict[flag]
        self.len = len(self.data)

    def __getitem__(self, idx):
        # Max-min regularization
        # data_item = self.data[idx]
        # data_tensor = torch.FloatTensor(data_item[:-1])
        # data_tensor = (data_tensor - torch.tensor([0.5])) / torch.tensor([0.5])
        # data_tensor = data_tensor.unsqueeze(0)
        # data_label = int(data_item[-1])

        # 归一化
        data_item = self.data[idx]
        data_tensor = torch.FloatTensor(data_item[:-1]).unsqueeze(0)
        # data_tensor = (data_tensor - self.box_cox_min) / (self.box_cox_max - self.box_cox_min)
        data_label = int(data_item[-1])

        return data_tensor, data_label

    def __len__(self):
        return self.len



class TestDataset(Dataset):
    def __init__(self, new_data_len, num_classes, feature_nums):
        super(TestDataset, self).__init__()
        self.feature_nums = feature_nums
        self.label_list = [i for i in range(num_classes)] * new_data_len
        random.shuffle(self.label_list)
        self.len = len(self.label_list)

    def __getitem__(self, index):
        return torch.randn(1, self.feature_nums), self.label_list[index]

    def __len__(self):
        return self.len


def get_loader():
    configs = getConfig()
    new_data_len, num_classes, feature_nums = configs['new_data_len'], configs['num_classes'], configs['feature_nums']

    train_dataset = CommonDataset(flag='train')
    valid_dataset = TestDataset(new_data_len, num_classes, feature_nums)
    test_dataset = TestDataset(new_data_len, num_classes, feature_nums)

    train_loader_configs = configs['train_loader']
    valid_loader_configs = configs['valid_loader']
    test_loader_configs = configs['test_loader']

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_loader_configs['bs'],
        shuffle=train_loader_configs['shuffle'],
        drop_last=train_loader_configs['drop_last'],
        num_workers=train_loader_configs['workers']
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_loader_configs['bs'],
        shuffle=valid_loader_configs['shuffle'],
        drop_last=valid_loader_configs['drop_last'],
        num_workers=valid_loader_configs['workers']
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_loader_configs['bs'],
        shuffle=test_loader_configs['shuffle'],
        drop_last=test_loader_configs['drop_last'],
        num_workers=test_loader_configs['workers']
    )

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_loader()
    for idx, (data, label) in enumerate(train_loader):
        print(" ------------ ")
        print(data.shape)
        # print(label.shape)
        # print(label)
