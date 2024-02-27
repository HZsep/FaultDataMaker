import torch
import random
import numpy as np
import csv

from openpyxl import Workbook

from dataset import CommonDataset
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter(data_element):
    # if 48.0 < data_element[0][0] < 51.0 and 25.0 < data_element[0][39] < 46.0 and 14.0 < data_element[0][-1] < 19.0:
    if 48.0 < data_element[0][0] < 51.0:
        return data_element[0]
    else:
        return -1

def get_ori_data():
    ori_dataset = CommonDataset(flag='train')
    loader = DataLoader(
        dataset=ori_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    data_dict = {}
    for class_idx in range(1, 21):
        data_dict[class_idx] = []
    for d, label in loader:
        label_name = label.item()+1
        t = d.squeeze(0).numpy().tolist()[0]
        t.append(label_name)
        data_dict[label_name].append(t)
    # print(data_dict[1][0])
    # print(len(data_dict[1]))
    return data_dict



def data_maker(model_path, data_num, save_path=None):
    set_seed(42)
    data_dicts_ori = get_ori_data()
    model = torch.load(model_path).cuda()
    model.eval()
    class_label = [i for i in range(1, 21)]

    count_idx = 0
    with torch.no_grad():
        for class_name in class_label:
            cls_data = []
            class_input = class_name - 1

            for i in range(100000):
                label = torch.tensor([class_input]).cuda()
                random_data = torch.randn(4096 * 8,  1, 41).cuda()
                output_data = model(random_data, label).cpu().detach().numpy().tolist()
                for out_data in output_data:
                    out_data_filter = filter(out_data)
                    if out_data_filter == -1:
                        continue
                    else:
                        if count_idx < data_num:
                            out_data_filter.append(class_name)
                            count_idx += 1
                            # print("count_idx:", count_idx)
                            cls_data.append(out_data_filter)
                        else:
                            data_dicts_ori[class_name].extend(cls_data)
                            break
                if count_idx >= data_num:
                    count_idx = 0
                    break
    total_data = []
    for key, value in data_dicts_ori.items():
        total_data.extend(value)
    print(" ----------------------- Done -----------------------")
    return total_data

def csv_wirter(data):
    wb = Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)
    wb.save("./data/new_data.xlsx")



if __name__ == '__main__':
    data_num = 400
    model_path = './save/best.pth'
    total_data = data_maker(model_path=model_path, data_num=data_num)
    print("sample len: ", len(total_data))
    # # print(len(total_data))
    csv_wirter(total_data)


