import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from sklearn import preprocessing
from tqdm import tqdm


class Evaluation():
    def __init__(self, datatype, runtype=''):
        if datatype == 'VOC':
            self.true_list = None
            if runtype == 'RQ3':
                with open('./data/VOCsel_augGT.json') as f:
                    self.true_list = json.load(f)
            else:
                with open('./data/VOCvalGT.json') as f:
                    self.true_list = json.load(f)
            self.id2gtbbx = {x['image_id']: x['bbox'] for x in self.true_list}
            self.id2gtlabel = {x['image_id']: x['category_id'] for x in self.true_list}
            self.id2prebbx = {x['image_id']: [] for x in self.true_list}
            self.id2prelabel = {x['image_id']: [] for x in self.true_list}
            self.id2dfbbx = {x['image_id']: [] for x in self.true_list}
            self.imgid_list = sorted([x['image_id'] for x in self.true_list])

        if datatype == 'COCO':
            with open('./data/instances_val2017.json') as f:
                self.true_list = json.load(f)
            self.GT_list = self.true_list["annotations"]
            self.id2gtbbx = {x['id']: [] for x in self.true_list['images']}
            self.id2gtlabel = {x['id']: [] for x in self.true_list['images']}
            self.id2prebbx = {x['id']: [] for x in self.true_list['images']}
            self.id2prelabel = {x['id']: [] for x in self.true_list['images']}
            self.id2dfbbx = {x['id']: [] for x in self.true_list['images']}
            self.imgid_list = sorted([x['id'] for x in self.true_list['images']])
            for x in self.GT_list:
                self.id2gtbbx[x['image_id']] = self.id2gtbbx[x['image_id']] + [self.transxyxy(x['bbox'])]
                self.id2gtlabel[x['image_id']] = self.id2gtlabel[x['image_id']] + [x['category_id']]

    def get_iou(self, X, Y):
        if Y == []:
            return torch.Tensor([[0] for _ in range(len(X))])
        return box_ops.box_iou(torch.tensor(X), torch.tensor(Y))

    def check_(self, pre_list, id2prebox):
        id_list = np.array(sorted([key if id2prebox[key] != [] else -1 for key in id2prebox]))
        id_list = id_list[np.where(id_list != -1)]
        ind = 0
        ind_ = {id_: 0 for id_ in id_list}
        for i in range(len(pre_list)):
            if i != 0 and pre_list[i]['image_id'] == pre_list[i - 1]['image_id']:
                continue
            assert pre_list[i]['image_id'] == id_list[ind]
            assert self.transxyxy(pre_list[i]['bbox']) == id2prebox[pre_list[i]['image_id']][
                ind_[pre_list[i]['image_id']]]
            ind_[pre_list[i]['image_id']] += 1
            ind += 1

    def get_dif(self, pre_list, dif_list):

        for x in pre_list:
            self.id2prebbx[x['image_id']] = self.id2prebbx[x['image_id']] + [self.transxyxy(x['bbox'])]
        for x in dif_list:
            self.id2dfbbx[x['image_id']] = self.id2dfbbx[x['image_id']] + [self.transxyxy(x['bbox'])]
        self.check_(pre_list, self.id2prebbx)
        self.check_(dif_list, self.id2dfbbx)
        ious_list = []
        for img_id in self.imgid_list:
            if self.id2prebbx[img_id] == []:
                continue
            ious = self.get_iou(self.id2prebbx[img_id], self.id2dfbbx[img_id])
            ious = torch.max(ious, dim=1).values.numpy().tolist()
            ious_list = ious_list + ious
        assert len(ious_list) == len(pre_list)
        return ious_list

    def get_score_iou(self, pre_list):
        for x in pre_list:
            self.id2prebbx[x['image_id']] = self.id2prebbx[x['image_id']] + [self.transxyxy(x['bbox'])]

        self.check_(pre_list, self.id2prebbx)
        ious_list = []
        gt_list = []
        tp_list = []
        for img_id in self.imgid_list:
            if self.id2prebbx[img_id] == []:
                continue
            ious = self.get_iou(self.id2prebbx[img_id], self.id2gtbbx[img_id])
            iousval = torch.max(ious, dim=1).values.numpy().tolist()
            ious_list = ious_list + iousval
            inds = torch.max(ious, dim=1).indices.numpy()
            gtlabel = [0 if len(self.id2gtlabel[img_id]) == 0 else self.id2gtlabel[img_id][inds[i]] for i in
                       range(len(inds))]
            gt_list = gt_list + gtlabel
        score_list = [pre_list[i]['full_score'][gt_list[i]] for i in range(len(pre_list))]
        tp_list = [0 if ious_list[i] > 0.5 and gt_list[i] == pre_list[i]['category_id'] else 1 for i in
                   range(len(pre_list))]
        assert len(score_list) == len(ious_list)
        return score_list, ious_list, tp_list

    def get_loss(self, pre_list):

        for x in pre_list:
            self.id2prebbx[x['image_id']] = self.id2prebbx[x['image_id']] + [self.transxyxy(x['bbox'])]

        self.check_(pre_list, self.id2prebbx)
        ious_list = []
        gt_list = []
        for img_id in self.imgid_list:
            if self.id2prebbx[img_id] == []:
                continue
            ious = self.get_iou(self.id2prebbx[img_id], self.id2gtbbx[img_id])
            iousval = torch.max(ious, dim=1).values.numpy().tolist()
            ious_list = ious_list + iousval
            inds = torch.max(ious, dim=1).indices.numpy()
            gtlabel = [0 if len(self.id2gtlabel[img_id]) == 0 else self.id2gtlabel[img_id][inds[i]] for i in
                       range(len(inds))]
            gt_list = gt_list + gtlabel

        assert len(ious_list) == len(pre_list)
        all_loss = []
        for i, x in enumerate(pre_list):
            cls_loss = F.cross_entropy(torch.tensor(x['full_score']), torch.tensor(gt_list[i])).item()
            all_loss.append([cls_loss, 1 - ious_list[i]])

        # return all_loss
        min_max_scaler = preprocessing.MinMaxScaler()
        all_loss = min_max_scaler.fit_transform(all_loss)

        return [x[0] + x[1] for x in all_loss]

    def get_faulttype(self, pre_list):
        for x in pre_list:
            self.id2prebbx[x['image_id']] = self.id2prebbx[x['image_id']] + [self.transxyxy(x['bbox'])]
            self.id2prelabel[x['image_id']] = self.id2prelabel[x['image_id']] + [x['category_id']]
        self.check_(pre_list, self.id2prebbx)
        fault_list = []
        faults = {}
        for img_id in self.imgid_list:
            if self.id2prebbx[img_id] == []:
                continue
            ious = self.get_iou(self.id2prebbx[img_id], self.id2gtbbx[img_id])
            iousval = torch.max(ious, dim=1).values.numpy()
            inds = torch.max(ious, dim=1).indices.numpy()
            prelabel = self.id2prelabel[img_id]
            gtlabel = [
                -1 if inds[i] >= len(self.id2gtlabel[img_id]) or iousval[i] < 0.5 else self.id2gtlabel[img_id][inds[i]]
                for
                i in range(len(inds))]
            for i in range(len(prelabel)):
                if prelabel[i] != gtlabel[i]:
                    if faults.get((gtlabel[i], prelabel[i]), -1) == -1:
                        faults[(gtlabel[i], prelabel[i])] = 1
                        fault_list.append((gtlabel[i], prelabel[i]))
                    else:
                        faults[(gtlabel[i], prelabel[i])] = faults[(gtlabel[i], prelabel[i])] + 1
                        fault_list.append((gtlabel[i], prelabel[i]))
                else:
                    fault_list.append((gtlabel[i], prelabel[i]))
        assert len(fault_list) == len(pre_list)
        # print(len(faults))
        # print(faults)
        # plt.bar([_ for _ in range(len(faults))], list(faults.values()))
        # plt.show()
        return fault_list, len(faults)

    def transxyxy(self, bbx):
        return [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]

    def RAUC_reg(self, metriclosslist):
        gtlosslist = sorted(metriclosslist)[::-1]
        for i in range(1, len(gtlosslist)):
            gtlosslist[i] = gtlosslist[i] + gtlosslist[i - 1]
            metriclosslist[i] = metriclosslist[i] + metriclosslist[i - 1]
        return sum(metriclosslist[:500]) / sum(gtlosslist[:500]), sum(metriclosslist[:1000]) / sum(
            gtlosslist[:1000]), sum(metriclosslist[:2000]) / sum(gtlosslist[:2000]), sum(metriclosslist[:5000]) / sum(
            gtlosslist[:5000]), sum(metriclosslist) / sum(gtlosslist)

    def RAUC_cls(self, metric_tplist):
        gt_tplist = sorted(metric_tplist)[::-1]
        x_list = [0]
        for i in range(1, len(gt_tplist)):
            gt_tplist[i] = gt_tplist[i] + gt_tplist[i - 1]
            metric_tplist[i] = metric_tplist[i] + metric_tplist[i - 1]
            x_list.append(i)
        return x_list, gt_tplist, metric_tplist, sum(metric_tplist[:500]) / sum(gt_tplist[:500]), sum(
            metric_tplist[:1000]) / sum(
            gt_tplist[:1000]), sum(metric_tplist[:2000]) / sum(gt_tplist[:2000]), sum(metric_tplist[:5000]) / sum(
            gt_tplist[:5000]), sum(metric_tplist) / sum(gt_tplist)


# if __name__ == "__main__":
#     test = Evaluation()
#     with open('instances_results065.json') as f1:
#         pre_list = json.load(f1)
#     print(test.get_loss(pre_list))
#     # loss = F.cross_entropy(torch.tensor([[1., 0, 0, 0, 0]]), torch.tensor([0])).item()
#     # print(loss)
