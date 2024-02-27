import pandas as pd
import numpy as np
from utils.utils import getConfig, json_loader, save2json
from scipy import stats
from scipy.special import inv_boxcox


# Reads an xlsx file and returns a list of dictionaries.
def data_creator():
    configs = getConfig()
    total_set, data_len = xlsx_reader(configs=configs)
    train_set, valid_set = split(total_set, data_len, configs)
    statistics = get_statistics([train_set, valid_set])
    data_normalized, statistics = normalize(datasets=[train_set, valid_set], statistics=statistics, configs=configs)
    train_dataset, valid_dataset = data_normalized
    save2json(dict_data=statistics, file_path=configs['ori_data_stat_path'])
    # return train_normal_dataset, valid_normal_dataset
    return train_dataset, valid_dataset


# Read the file
def xlsx_reader(configs):
    xlsx_path, data_len, num_classes = configs['ori_data_path'], configs['ori_data_len'], configs['num_classes']
    total_dataset, total_data = {}, []
    for class_name in range(num_classes):
        total_dataset[class_name] = []
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    for index, row in df.iterrows():
        label = int(list(row.to_dict().values())[-1]) - 1
        if len(total_dataset[label]) < data_len:
            total_dataset[label].append(list(row.to_dict().values()))  # 包含label的数据

    return total_dataset, data_len


# Data set split
def split(total_dataset, data_len, configs):
    train_ratio = configs['train_ratio']
    valid_ratio = 1 - train_ratio
    train_len, valid_len = int(data_len * train_ratio), int(data_len * valid_ratio)
    train_set, valid_set = [], []
    for key in total_dataset.keys():
        for idx, value in enumerate(total_dataset[key]):
            data, label = value[:-1], value[-1] - 1
            # tmp = np.array(data)
            data.append(label)
            # if np.min(tmp) > 0:
            if train_len > idx >= 0:
                train_set.append(data)
            if valid_len + train_len >= idx >= train_len:
                valid_set.append(data)
    return train_set, valid_set


# Getting the mean and mean square of the data
def get_statistics(datasets: list):
    data = []
    for dataset in datasets:
        for data_line in dataset:
            data.append(data_line[:-1])
    data = np.array(data)
    statistics = {"std": np.std(data), "mean": np.mean(data), "max": np.max(data), "min": np.min(data)}
    return statistics


# Standardized data pre-processing
def normalize(datasets, statistics, configs):
    normal_seq = configs['normal_seq']
    total_datas, total_label = [], []
    train_len, valid_len = len(datasets[0]), len(datasets[1])
    for _, dataset in enumerate(datasets):
        for data_line in dataset:
            data_infor, data_label, flag = data_line[:-1], data_line[-1], 'train'
            total_datas.append(data_infor)
            total_label.append(data_label)
    # Normalization and normalization requires data to be transformed into 1-dimensional
    total_datas = np.array(total_datas)
    shape = total_datas.shape
    total_datas = total_datas.reshape(-1)
    for normal in normal_seq:
        normal_data, statistics = normal_list[normal](data=total_datas, statistics=statistics)
        total_datas = normal_data
    total_datas = total_datas.reshape(shape)
    # train_datas, valid_datas = total_datas[:train_len], total_datas[train_len:train_len+valid_len]
    total_datas = total_datas.tolist()
    train_dataset, valid_dataset = [], []
    for idx in range(len(total_label)):
        data_line, label_for_idx = total_datas[idx], total_label[idx]
        data_line.append(label_for_idx)
        if idx < train_len:
            train_dataset.append(data_line)
        else:
            valid_dataset.append(data_line)
    data_norm = (train_dataset, valid_dataset)
    return data_norm, statistics


# Max-min normalization
def max_min_normal(**kwargs):
    data, statistics = kwargs['data'], kwargs['statistics']
    data = np.array(data)
    max_num, min_num = np.max(data), np.min(data)
    noraml_data = (data - min_num) / (max_num - min_num)
    statistics['max_min_normal_before_max'] = max_num
    statistics['max_min_normal_before_min'] = min_num
    return noraml_data, statistics


def max_normal(**kwargs):
    data, statistics = kwargs['data'], kwargs['statistics']
    data = np.array(data)
    max_num = np.max(data)
    noraml_data = data / max_num
    statistics['max_normal_before_max'] = max_num
    return noraml_data, statistics


# box-cox regularization
def box_cox_normal(**kwargs):
    data, statistics = kwargs['data'], kwargs['statistics']
    translation_eps = 1e-5
    translation_factor = np.abs(np.min(data)) + translation_eps
    box_cox_data = np.array(data) + translation_factor
    shape = box_cox_data.shape
    # print(np.min(box_cox_data))
    box_cox_data = box_cox_data.reshape(-1)
    noraml_data, lambda_params = stats.boxcox(box_cox_data)
    noraml_data = noraml_data.reshape(shape)
    statistics['lambda_params'] = lambda_params
    return noraml_data, statistics


# def box_split_normal(**kwargs):
#     data = kwargs['data']
#     for

def mean_std_normal(**kwargs):
    data, statistics = kwargs['data'], kwargs['statistics']
    data = np.array(data)
    mean, std = np.mean(data), np.std(data)
    # print(np.min(box_cox_data))
    noraml_data = (data - mean) / std
    statistics['mean_std_normal_before_mean'] = mean
    statistics['mean_std_normal_before_std'] = std
    return noraml_data, statistics


def inverse(data, configs):
    # print(configs['normal_seq'])
    normals = configs['normal_seq']
    for normal in reversed(normals):
        # print(normal)
        inverse_data = inverse_list[normal](data=data, configs=configs)
        data = inverse_data
    return data


def max_min_inverse(**kwargs):
    data, configs = kwargs['data'], kwargs['configs']
    json_path = configs['ori_data_stat_path']
    params = json_loader(json_path)
    max_min_normal_after_max = params['max_min_normal_before_max']
    max_min_normal_after_min = params['max_min_normal_before_min']
    inverse_data = np.array(data) * (max_min_normal_after_max - max_min_normal_after_min) + max_min_normal_after_min
    return inverse_data.tolist()


def max_inverse(**kwargs):
    data, configs = kwargs['data'], kwargs['configs']
    json_path = configs['ori_data_stat_path']
    params = json_loader(json_path)
    max_num = params['max_normal_before_max']
    inverse_data = np.array(data) * max_num
    return inverse_data


def box_cox_inverse(**kwargs):
    data, configs = kwargs['data'], kwargs['configs']
    json_path = configs['ori_data_stat_path']
    params = json_loader(json_path)
    lambda_params = params['lambda_params']
    inverse_data = inv_boxcox(data, lambda_params)
    return inverse_data


def mean_std_inverse(**kwargs):
    data, configs = kwargs['data'], kwargs['configs']
    json_path = configs['ori_data_stat_path']
    params = json_loader(json_path)
    data = np.array(data)
    mean = params['mean_std_normal_before_mean']
    std = params['mean_std_normal_before_std']
    inverse_data = data * std + mean
    return inverse_data


# 数据反标准化
def data_inverse(normal_data, std, mean):
    ori_data = normal_data * std + mean
    return ori_data


normal_list = {
    'max-min': max_min_normal,
    'box-cox': box_cox_normal,
    'mean-std': mean_std_normal,
    'max': max_normal
}

inverse_list = {
    'max-min': max_min_inverse,
    'box-cox': box_cox_inverse,
    'mean-std': mean_std_inverse,
    'max': max_inverse
}

if __name__ == "__main__":
    train, valid = data_creator()
    # d_t = []
    # for d in train:
    #     d_t.append(d[:-1])
    # print(np.max(d_t))
    # print(np.min(d_t))
    # print(np.mean(d_t))
    print(train[231])
    # for sample in train:
    #     print(sample)
    config = getConfig()
    # print(configs['normal_seq'][0])
    total_data_train = []
    for line in train:
        total_data_train.append(line[:-1])
    # print(total_data)
    inversed_data = inverse(data=total_data_train, configs=config)
    print(inversed_data[231])
