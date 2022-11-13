import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.io.image import read_image

import my_transforms
from Metrics import Metrics
from evaluation import Evaluation
from modelcodes.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from my_dataset import COCOdataset
import json

from visualization import visualization

device = "cuda"
coco_root = 'coco2017'
batchsize = 1


def Inference():
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
        with open('./data/FRCNN_COCO070.json', 'w') as json_file:
            json_file.write(json_str)


def DeepView(datapath):
    with open(datapath) as f1:
        pre_list = json.load(f1)
    eva = Evaluation(datatype='COCO')
    eva2 = Evaluation(datatype='COCO')
    fault_list, fault_type_num = eva.get_faulttype(pre_list)
    _, _, tp_list = eva2.get_score_iou(pre_list)
    result = {}
    metrics = Metrics(pre_list=pre_list, dif_list=None, imgpre_list=None)
    result['DeepView'] = metrics.deepview('COCO')
    metric_names = [_[0] for _ in result.items()]
    vis = visualization()
    colorlist = ['xkcd:red', 'xkcd:peach', 'xkcd:green', 'xkcd:light purple', 'xkcd:black', 'xkcd:grey']

    deepview_instance = None
    X = []
    Tro_diversity = []
    for i in range(len(result)):
        metric = result[metric_names[i]]
        zipped = zip(metric, fault_list, tp_list, pre_list)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        srt_ = zip(*sort_zipped)
        srtd_metric_list, srtd_fault_list, srtd_tp_list, srtd_pre_list = [list(x) for x in srt_]
        deepview_instance = srtd_pre_list[::-1]
        # rq3 = RQ3engine(slice=[0.1])
        # rq3.map2image(srtd_pre_list[::-1], metric_names[i], modeltype)
        X, Y, Tro_diversity = vis.plt_fault_type_rauc(srtd_fault_list[::-1], fault_type_num)
        plt.plot(X, Y, label=metric_names[i], color=colorlist[i])
        x_list, Tro_effective, y_list, rauc_1, rauc_2, rauc_3, rauc_5, rauc_all = eva.RAUC_cls(
            metric_tplist=srtd_tp_list[::-1])
        print(metric_names[i] + ' diversity: RAUC =' + ' ' + str(round(sum(Y) / sum(Tro_diversity) * 100, 2)))
        # plt.plot(x_list, y_list, label=metric_names[i])
        print(metric_names[i] + ' effectiveness: RAUC-n = ' + str(
            [round(rauc_1 * 100, 2), round(rauc_2 * 100, 2), round(rauc_3 * 100, 2), round(rauc_5 * 100, 2),
             round(rauc_all * 100, 2)]))

    plt.plot(X, Tro_diversity, label='Theoretical', color='tab:blue')

    plt.xlabel('$Number\ of\ prioritized\ test\ instances$')
    plt.ylabel('$Number\ of\ error\ type\ detected$')

    plt.legend()

    plt.show()

    return deepview_instance


if __name__ == "__main__":
    # Inference()  # Model Inference

    deepview_result = DeepView(datapath='./data/FRCNN_COCO070.json')

    json_str = json.dumps(deepview_result, indent=4)
    with open('./data/deepview_result', 'w') as json_file:
        json_file.write(json_str)
