import numpy as np
import torch
import my_transforms
from torchvision.io.image import read_image
from modelcodes.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision
from tqdm import tqdm
from my_dataset import COCOdataset, VOCDataSet, RQ3VOCDataSet
import json
from torchvision import transforms
import transforms2
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

device = "cuda"
coco_root = 'coco2017'
VOC_root = './VOCdevkit'
batchsize = 1


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# 转为: x_min, y_min, w, h (COCO格式)
def transxywh(bbx):
    return [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]]


def RUN(T, isgaus, runtype):
    ex_transforms = torch.nn.Sequential(
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=2.5)
    )

    # weights = SSD300_VGG16_Weights.DEFAULT

    print('T = ' + str(int(T) / 100))
    print('isgaus = ' + str(isgaus))
    print('is_RQ3 = '+str(runtype=='RQ3'))
    model = fasterrcnn_resnet50_fpn_v2(num_classes=21, box_score_thresh=int(T) / 100)  #
    checkpoint = torch.load('./models/FRCNN_VOC_orgmodel_epoch3.pth', map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    data_transform = {
        "val": transforms2.Compose([transforms2.ToTensor()])
    }

    val_dataset = None
    if isgaus:
        if runtype == 'RQ3':
            val_dataset = RQ3VOCDataSet(VOC_root, "2012", data_transform['val'], "sel_aug.txt",
                                        ex_transforms=my_transforms.AddGaussianNoise(amplitude=10))
        else:
            val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "val.txt",
                                     ex_transforms=my_transforms.AddGaussianNoise(amplitude=10))
    else:
        if runtype == 'RQ3':
            val_dataset = RQ3VOCDataSet(VOC_root, "2012", data_transform['val'], "sel_aug.txt", ex_transforms=None)
        else:
            val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "val.txt", ex_transforms=None)
    # ex_transforms=my_transforms.AddGaussianNoise(amplitude=10)

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

    coco = get_coco_api_from_dataset(val_dataset_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            image = list(img.to(device) for img in image)

            outputs = model(image)

            cpu_device = torch.device("cpu")

            # evaluation
            eva_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, eva_outputs)}
            coco_evaluator.update(res)

            for i, prediction in enumerate(outputs):
                cat_ids = prediction["labels"].cpu()
                bboxs = prediction["boxes"].cpu().numpy().tolist()
                scores = prediction['scores'].cpu()
                dis_scores = prediction['dis_scores'].cpu().numpy().tolist()
                full_score = prediction['full_scores'].cpu().numpy().tolist()
                for j in range(prediction["labels"].shape[0]):
                    content_dic = {
                        "image_path": targets[i]["image_path"],
                        "image_id": int(targets[i]["image_id"].numpy()[0]),
                        "category_id": int(cat_ids[j]),
                        "bbox": transxywh(bboxs[j]),
                        "score": float(scores[j]),
                        "full_score": full_score[j],
                        "dis_score": dis_scores[j]
                    }
                    results.append(content_dic)
            break
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        json_str = json.dumps(results, indent=4)
        if isgaus:
            with open(runtype + 'gaus_FRCNN_VOC' + T + '.json', 'w') as json_file:
                json_file.write(json_str)
        else:
            with open(runtype + 'FRCNN_VOC' + T + '.json', 'w') as json_file:
                json_file.write(json_str)


if __name__ == "__main__":
    gaus_list = {False, True}
    for gaus in gaus_list:
        RUN(T='070', isgaus=gaus, runtype='RQ3')
    # t_list = {'070', '080', '090'}
    # gaus_list = {False, True}
    # for t in t_list:
    #     for gaus in gaus_list:
    #         RUN(T=t, isgaus=gaus)
