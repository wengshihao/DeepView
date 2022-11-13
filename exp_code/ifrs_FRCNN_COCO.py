import numpy as np
import torch
from torchvision.io.image import read_image

import my_transforms
from modelcodes.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from my_dataset import COCOdataset
import json

device = "cuda"
coco_root = 'coco2017'
batchsize = 1


# 转为: x_min, y_min, w, h (COCO格式)
def transxywh(bbx):
    bbx[2] = bbx[2] - bbx[0]
    bbx[3] = bbx[3] - bbx[1]
    return bbx


weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)

data_transform = weights.transforms()

val_dataset = COCOdataset(coco_root, "val", data_transform)  #

val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batchsize,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=val_dataset.collate_fn)
model.to(device)
model.eval()
results = []
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
    with open('gaus_FRCNN_COCO070.json', 'w') as json_file:
        json_file.write(json_str)


