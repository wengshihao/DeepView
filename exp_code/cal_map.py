import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# accumulate predictions from all images

coco_true = COCO(annotation_file="coco2017/annotations/instances_val2017.json")

coco_pre = coco_true.loadRes('t2_SSD_instances_results050.json')

coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()


# json_path = 'instances_val2017.json'
# img_path = 'val2017'
# coco = COCO(annotation_file=json_path)
# ids = list(sorted(coco.imgs.keys()))
# print("number of images: {}".format(len(ids)))
#
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
