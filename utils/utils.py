import yaml
import json


def getConfig():
    with open('./config.yaml', "r", encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save2json(dict_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(obj=dict_data, fp=file)


def json_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data
