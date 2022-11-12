import math
import os
import sys

import torch
from torchvision.transforms import transforms
import transforms2
import presets
import transforms as T
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import torchvision
from eva_utils import get_coco_api_from_dataset, CocoEvaluator
from modelcodes.models.detection import _utils as det_utils
from my_dataset import VOCDataSet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
import utils
from engine import evaluate, train_one_epoch

resume = None
output_dir = './models'
VOC_root = './VOCdevkit'
device = 'cuda'
batch_size = 4
epochs = 10
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
data_transform = {
    "val": transforms2.Compose([transforms2.ToTensor()])
}
train_dataset = VOCDataSet(VOC_root, "2012", presets.DetectionPresetTrain(data_augmentation='ssd'), "train.txt")

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=0,
                                                collate_fn=train_dataset.collate_fn)

val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "val.txt")
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0,
                                                  collate_fn=val_dataset.collate_fn)

# replace the pre-trained head with a new one

model.head.classification_head = SSDClassificationHead([512, 1024, 512, 256, 256, 256],
                                                       model.anchor_generator.num_anchors_per_location(), 21)

optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.002,
                            momentum=0.9,
                            weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
model.to(device)
cpu_device = torch.device("cpu")

start_epoch = 0

if resume is not None:
    print('In resume training.')
    checkpoint = torch.load(resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"] + 1

for epoch in range(start_epoch, epochs):
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, 10)
    lr_scheduler.step()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    utils.save_on_master(checkpoint, os.path.join(output_dir, f"SSD_VOC_orgmodel_{epoch}.pth"))

    evaluate(model, val_data_set_loader, device=device)

# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
