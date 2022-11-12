import json
import os

import torch
from torchvision.ops import boxes as box_ops


class RQ3engine():
    def __init__(self, slice):
        self.slice = slice
        self.instance_num = 15614

        with open('./data/VOCsel_augGT.json') as f:
            self.true_list = json.load(f)
        self.id2gtbbx = {x['image_id']: x['bbox'] for x in self.true_list}
        self.id2gtlabel = {x['image_id']: x['category_id'] for x in self.true_list}
        self.id2gtarea = {x['image_id']: x['area'] for x in self.true_list}

        with open('./data/VOCtrainGT.json') as f:
            self.train_list = json.load(f)

    def transxyxy(self, bbx):
        return [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]

    def get_iou(self, X, Y):
        if Y == []:
            return torch.Tensor([[0] for _ in range(len(X))])
        return box_ops.box_iou(torch.tensor(X), torch.tensor(Y))

    def writeforretrain(self, metric_name, modeltype, ratio, map_image_dict):
        with open("./VOCdevkit/VOC2012/ImageSets/Retrain/" + metric_name + '_' + modeltype + '_' + str(
                ratio * 100) + '.txt', "w") as f:
            # write original train set
            for x in self.train_list:
                f.write(x['image_path'][:-4] + '\n')
            # write extra train set
            for x in map_image_dict:
                f.write(map_image_dict[x]['image_path'][:-4] + '\n')

        # write original and extra train annotation
        path = './VOCdevkit/VOC2012/Retrain_Annotation/' + metric_name + '_' + modeltype + '_' + str(ratio * 100)
        assert os.path.exists(path) == False
        os.makedirs(path)
        # write original
        for x in self.train_list:
            tmp = {
                'image_path': x['image_path'],
                'category_id': x['category_id'],
                'bbox': x['bbox'],
                'area': x['area']
            }
            json_str = json.dumps(tmp, indent=4)
            with open(path + '/' + x['image_path'][:-4] + '.json', 'w') as json_file:
                json_file.write(json_str)
        # write extra
        for x in map_image_dict:
            tmp = {
                'image_path': map_image_dict[x]['image_path'],
                'category_id': map_image_dict[x]['category_id'],
                'bbox': map_image_dict[x]['bbox'],
                'area': map_image_dict[x]['area']
            }
            json_str = json.dumps(tmp, indent=4)
            with open(path + '/' + map_image_dict[x]['image_path'][:-4] + '.json', 'w') as json_file:
                json_file.write(json_str)

    def inslvlgen(self, srt_prelist, ratio, id2preboxdict, id2prelabeldict, metric_name, modeltype):
        ins_map_num = int(ratio * self.instance_num)
        map_prelist = srt_prelist[:ins_map_num]
        map_image_dict = {}

        # get predict dict
        for x in map_prelist:
            boxes = self.transxyxy(x['bbox'])
            area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
            if map_image_dict.get(x['image_id']) == None:
                map_image_dict[x['image_id']] = {'image_path': x['image_path'],
                                                 'image_id': x['image_id'],
                                                 'category_id': [x['category_id']],
                                                 'bbox': [boxes],
                                                 'area': [area]}
            else:
                assert x['image_id'] == map_image_dict[x['image_id']]['image_id']
                map_image_dict[x['image_id']]['category_id'] = map_image_dict[x['image_id']]['category_id'] + [
                    x['category_id']]
                map_image_dict[x['image_id']]['bbox'] = map_image_dict[x['image_id']]['bbox'] + [boxes]
                map_image_dict[x['image_id']]['area'] = map_image_dict[x['image_id']]['area'] + [area]

        # use ground truth to modify and add more instances
        for img_id in map_image_dict:
            ious = self.get_iou(map_image_dict[img_id]['bbox'], self.id2gtbbx[img_id])
            inds = torch.max(ious, dim=1).indices.tolist()

            map_image_dict[img_id]['category_id'] = [self.id2gtlabel[img_id][ind] for ind in inds]
            map_image_dict[img_id]['bbox'] = [self.id2gtbbx[img_id][ind] for ind in inds]
            map_image_dict[img_id]['area'] = [self.id2gtarea[img_id][ind] for ind in inds]

            prenum = int(len(id2preboxdict[img_id]))
            addnum = min((prenum - len(map_image_dict[img_id]['category_id'])), int(prenum / 3))
            if addnum != 0:
                boxes = [self.transxyxy(x) for x in id2preboxdict[img_id][(prenum - addnum):]]
                labels = [x for x in id2prelabeldict[img_id][(prenum - addnum):]]
                areas = [(x[3] - x[1]) * (x[2] - x[0]) for x in boxes]
                map_image_dict[img_id]['category_id'] = map_image_dict[img_id]['category_id'] + labels
                map_image_dict[img_id]['bbox'] = map_image_dict[img_id]['bbox'] + boxes
                map_image_dict[img_id]['area'] = map_image_dict[img_id]['area'] + areas
        print(metric_name + ' add image num = ' + str(len(map_image_dict)))
        self.writeforretrain(metric_name=metric_name, modeltype=modeltype, ratio=ratio,
                             map_image_dict=map_image_dict)

    def imglvlgen(self, srt_prelist, ratio, metric_name, modeltype):
        ins_map_num = int(ratio * self.instance_num)
        map_prelist = srt_prelist[:ins_map_num]
        map_image_dict = {}
        instancenum = 0
        barget = int(self.instance_num * ratio)
        for x in map_prelist:
            if map_image_dict.get(x['image_id']) == None and instancenum + len(
                    self.id2gtlabel[x['image_id']]) <= barget:  # 只有为None的时候才添加，避免图片重复添加

                map_image_dict[x['image_id']] = {'image_path': x['image_path'],
                                                 'image_id': x['image_id'],
                                                 'category_id': [],
                                                 'bbox': [],
                                                 'area': []}
                map_image_dict[x['image_id']]['category_id'] = self.id2gtlabel[x['image_id']]
                map_image_dict[x['image_id']]['bbox'] = self.id2gtbbx[x['image_id']]
                map_image_dict[x['image_id']]['area'] = self.id2gtarea[x['image_id']]
                instancenum += len(self.id2gtlabel[x['image_id']])
        print(metric_name + ' add image num = ' + str(len(map_image_dict)))
        self.writeforretrain(metric_name=metric_name, modeltype=modeltype, ratio=ratio,
                             map_image_dict=map_image_dict)

    def map2image(self, srt_prelist, metric_name, modeltype):
        id2preboxdict = {}
        id2prelabeldict = {}
        for x in srt_prelist:
            if id2preboxdict.get(x['image_id']) == None:
                id2preboxdict[x['image_id']] = [x['bbox']]
                id2prelabeldict[x['image_id']] = [x['category_id']]
            else:
                id2preboxdict[x['image_id']] = id2preboxdict[x['image_id']] + [x['bbox']]
                id2prelabeldict[x['image_id']] = id2prelabeldict[x['image_id']] + [x['category_id']]

        for ratio in self.slice:
            if metric_name[0:4] == 'ours' or metric_name == 'random-instance':
                print('gen ' + metric_name + 'ratio = ' + str(ratio))
                self.inslvlgen(srt_prelist, ratio, id2preboxdict, id2prelabeldict, metric_name, modeltype)
            else:
                print('gen ' + metric_name + 'ratio = ' + str(ratio))
                self.imglvlgen(srt_prelist, ratio, metric_name, modeltype)
