import json

import numpy as np
from matplotlib import pyplot as plt

from Metrics import Metrics
from evaluation import Evaluation
import seaborn as sns


class visualization():
    def plt_fault_type_rauc(self, srtd_fault_list, fault_type_num):
        fault_set = set()
        Y = []
        for x in srtd_fault_list:
            if x[0] == x[1]:
                Y.append(len(fault_set))
            else:
                fault_set.add(x)
                Y.append(len(fault_set))
        X = [_ for _ in range(len(srtd_fault_list))]
        Tro = [_ if _ < fault_type_num else fault_type_num for _ in range(len(srtd_fault_list))]
        return X, Y, Tro

    def draw_distribution(self, score_throd, modeltype, datatype, sel_num):
        with open('./data/' + modeltype + '_' + datatype + score_throd + '.json') as f1:
            pre_list = json.load(f1)
        with open('./data/gaus_' + modeltype + '_' + datatype + score_throd + '.json') as f1:
            dif_list = json.load(f1)
        with open('./data/' + modeltype + '_' + datatype + 'image.json') as f2:
            imgpre_list = json.load(f2)

        eva = Evaluation()
        score_list, iou_list, tp_list = eva.get_score_iou(pre_list)
        result = {}
        metrics = Metrics(pre_list=pre_list, dif_list=dif_list, imgpre_list=imgpre_list)
        # result['ours(difference1)'] = metrics.difference_1()
        result['ours(difference)'] = metrics.difference()
        result['ours(receptive field)'] = metrics.Dis_p()
        result['$1-pall_{max}$(instance)'] = metrics.one_minus_pmax(part="all")
        # result['1vs2'] = metrics.one_vs_two(0)
        # result['entropy'] = metrics.entropy_image(0)
        # result['Gini'] = metrics.gini(0)
        # result['random'] = metrics.random_srt()
        metric_names = [_[0] for _ in result.items()]
        col = ['red', 'green', 'blue']
        for i in range(len(result)):
            metric = result[metric_names[i]]
            zipped = zip(metric, score_list, iou_list, tp_list)
            sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
            srt_ = zip(*sort_zipped)
            srtd_metric_list, srtd_score_list, srtd_iou_list, srtd_tp_list = [list(x) for x in srt_]
            srtd_score_list = srtd_score_list[::-1]
            srtd_iou_list = srtd_iou_list[::-1]
            srtd_tp_list = srtd_tp_list[::-1]
            print(sum(srtd_tp_list[:1000])/1000)
            data = {'x': srtd_iou_list[:1000], 'y': srtd_score_list[:1000]}
            sns.jointplot(data=data, x='x', y='y', ylim=(0, 1))
            plt.show()


if __name__ == "__main__":
    vis = visualization()
    vis.draw_distribution(score_throd='030', modeltype='SSD', datatype='COCO', sel_num=50)
