import numpy as np
import torch
import my_transforms
from torchvision.io.image import read_image
from modelcodes.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from my_dataset import COCOdataset,VOCDataSet
import json
from torchvision import transforms
import transforms2
device = "cuda"
coco_root = 'coco2017'
batchsize = 1


# 转为: x_min, y_min, w, h (COCO格式)
def transxywh(bbx):
    bbx[2] = bbx[2] - bbx[0]
    bbx[3] = bbx[3] - bbx[1]
    return bbx


ex_transforms = torch.nn.Sequential(
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=2.5)
)

weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights, score_thresh=0.8)

data_transform = weights.transforms()
val_dataset = COCOdataset(coco_root, "val", transforms=data_transform,
                          ex_transforms=my_transforms.AddGaussianNoise(amplitude=10))  #
val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batchsize,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=val_dataset.collate_fn)
model.to(device)
model.eval()
results = []
image_results = []
with torch.no_grad():
    for image, targets in tqdm(val_dataset_loader, desc="validation..."):
        image = list(img.to(device) for img in image)

        outputs = model(image)

        for i, prediction in enumerate(outputs):
            cat_ids = prediction["labels"].cpu()
            bboxs = prediction["boxes"].cpu().numpy().tolist()
            scores = prediction['scores'].cpu()
            dis_scores = prediction['dis_scores'].cpu().numpy().tolist()
            full_score = prediction['full_scores'].cpu().numpy().tolist()
            for j in range(prediction["labels"].shape[0]):
                content_dic = {
                    "image_id": int(targets[i]["image_id"].numpy()[0]),
                    "category_id": int(cat_ids[j]),
                    "bbox": transxywh(bboxs[j]),
                    "score": float(scores[j]),
                    "full_score": full_score[j],
                    "dis_score": dis_scores[j]
                }
                results.append(content_dic)
    json_str = json.dumps(results, indent=4)
    with open('gaus_SSD_COCO080.json', 'w') as json_file:
        json_file.write(json_str)
