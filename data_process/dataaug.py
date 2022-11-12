import os
import random

from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

import my_transforms

txt_path = os.path.join('../VOCdevkit/VOC2012/', "ImageSets", "Main", 'val.txt')
print(txt_path)
with open(txt_path) as read:
    img_name_list = [line.strip() for line in read.readlines()]
assert len(img_name_list) == 5823

# with open("sel_aug.txt", "w") as f:
#     for name in img_name_list[:2911]:
#         f.write(name + '\n')
#         f.write(name+'_aug\n')
# with open("val_aug.txt", "w") as f:
#     for name in img_name_list[2911:]:
#         f.write(name + '\n')
#         f.write(name + '_aug\n')
for img_name in tqdm(img_name_list):
    image = Image.open('../VOCdevkit/VOC2012/JPEGImages/' + img_name + '.jpg')
    aug_func1 = T.ColorJitter(brightness=.07, hue=.07, saturation=.07, contrast=.07)
    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=1)
    image = blurrer(aug_func1(image))
    image.save('../VOCdevkit/VOC2012/JPEGImages/' + img_name + '_aug.jpg')
