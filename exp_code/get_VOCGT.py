import math
import os
import sys

import torch
from torchvision.transforms import transforms
from tqdm import tqdm

import transforms2

from my_dataset import VOCDataSet, RQ3VOCDataSet
import json

resume = None
VOC_root = './VOCdevkit'
device = 'cuda'

data_transform = {
    "val": transforms2.Compose([transforms2.ToTensor()])
}

val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "train.txt")
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0,
                                                  collate_fn=val_dataset.collate_fn)
result = []
for _, targets in tqdm(val_data_set_loader):
    T = {
        'image_path': targets[0]['image_path'],
        'image_id':targets[0]['image_id'][0].item(),
        'category_id': targets[0]['labels'].numpy().tolist(),
        'bbox': targets[0]['boxes'].numpy().tolist(),
        'area': targets[0]["area"].numpy().tolist(),
        'iscrowd': targets[0]["iscrowd"].numpy().tolist()
    }

    result.append(T)
print(len(result))
json_str = json.dumps(result, indent=4)
with open('data/VOCtrainGT.json', 'w') as json_file:
    json_file.write(json_str)

'''RQ3_gt.json

resume = None
VOC_root = './VOCdevkit'
device = 'cuda'

data_transform = {
    "val": transforms2.Compose([transforms2.ToTensor()])
}

val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "val.txt")
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0,
                                                  collate_fn=val_dataset.collate_fn)
result = []
for _, targets in tqdm(val_data_set_loader):
    T = {
        'image_path': targets[0]['image_path'],
        'category_id': targets[0]['labels'].numpy().tolist(),
        'bbox': targets[0]['boxes'].numpy().tolist(),
        'area': targets[0]["area"].numpy().tolist(),
        'iscrowd': targets[0]["iscrowd"].numpy().tolist()
    }
    json_str = json.dumps(T, indent=4)
    with open('./VOCdevkit/VOC2012/aug_Annotation/' + targets[0]['image_path'][:-4] + '.json', 'w') as json_file:
        json_file.write(json_str)

    T2 = {
        'image_path': targets[0]['image_path'][:-4] + '_aug.jpg',
        'category_id': targets[0]['labels'].numpy().tolist(),
        'bbox': targets[0]['boxes'].numpy().tolist(),
        'area': targets[0]["area"].numpy().tolist(),
        'iscrowd': targets[0]["iscrowd"].numpy().tolist()
    }
    json_str = json.dumps(T2, indent=4)
    with open('./VOCdevkit/VOC2012/aug_Annotation/' + targets[0]['image_path'][:-4] + '_aug.json', 'w') as json_file:
        json_file.write(json_str)

# print(len(result))
# json_str = json.dumps(result, indent=4)
# with open('data/VOCvalGT.json', 'w') as json_file:
#     json_file.write(json_str)

'''
