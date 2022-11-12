import math
import os
import random

import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead

import transforms2
import presets
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, \
    SSD300_VGG16_Weights, ssd300_vgg16
from my_dataset import RQ3VOCDataSet, RetrainVOCDataset, VOCDataSet
import utils
from engine import evaluate, train_one_epoch


def RETRAIN(metricname,epoch_num):
    print('start training = ' + metricname)
    resume = None
    output_dir = './models'
    VOC_root = './VOCdevkit'
    device = 'cuda'
    batch_size = 4
    epochs = epoch_num

    data_transform = {
        "val": transforms2.Compose([transforms2.ToTensor()])
    }

    #     model = fasterrcnn_resnet50_fpn_v2(num_classes=21)  #
    #     checkpoint = torch.load('./models/FRCNN_VOC_orgmodel_epoch3.pth', map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])

    train_dataset = RetrainVOCDataset(VOC_root, "2012", presets.DetectionPresetTrain(data_augmentation='ssd'),
                                      metricname)
    # train_dataset = VOCDataSet(VOC_root, "2012", presets.DetectionPresetTrain(data_augmentation='hflip'), "train.txt")

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=0,
                                                    collate_fn=train_dataset.collate_fn)
    print('train set num = ' + str(len(train_data_loader) * batch_size))

    val_dataset = RQ3VOCDataSet(VOC_root, "2012", data_transform['val'], "val_aug.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=0,
                                                      collate_fn=val_dataset.collate_fn)
    print('val set num = ' + str(len(val_data_set_loader)))
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)
    model.head.classification_head = SSDClassificationHead([512, 1024, 512, 256, 256, 256],
                                                           model.anchor_generator.num_anchors_per_location(), 21)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
    model.to(device)
    cpu_device = torch.device("cpu")

    start_epoch = 1

    if resume is not None:
        print('In resume training.')
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, 30)
        lr_scheduler.step()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        utils.save_on_master(checkpoint, os.path.join(output_dir, metricname + f"_epoch{epoch}.pth"))

        eva = evaluate(model, val_data_set_loader, device=device)
        with open('retrain_results/' + metricname + '.txt', 'a') as f:
            f.write('epoch = ' + str(epoch) + ': ' + str(eva.res) + '\n')


if __name__ == "__main__":
    # metric_list = ['ours(difference)_FRCNN_2.5', 'ours(receptive field)_FRCNN_2.5', '1vs2_FRCNN_2.5',
    #                'entropy_FRCNN_2.5', 'Gini_FRCNN_2.5', 'random-instance_FRCNN_2.5',
    #                'random-image_FRCNN_2.5']
    # for metric_name in metric_list:
    RETRAIN()
# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
