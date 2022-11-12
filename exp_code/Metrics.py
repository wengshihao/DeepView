import math
import random
import xmlrpc.client

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from evaluation import Evaluation


class Metrics():
    def __init__(self, pre_list, dif_list, imgpre_list=None):
        self.pre_list = pre_list
        self.imgpre_list = imgpre_list
        self.dif_list = dif_list

    def Euler_distance(self):
        def getdis(x_list, y_list):
            dis = 0.
            for x, y in zip(x_list, y_list):
                dis += (x - y) ** 2
            return math.sqrt(dis)

        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gt = [0 for _ in range(len(full_score))]
            gt[full_score.index(max(full_score[1:]))] = 1
            assert gt[0] != 1
            result.append([x['dis_score'], getdis(gt, full_score)])
        return result

    def Euler_distance_DIS(self):
        def getdis(x_list, y_list):
            dis = 0.
            for x, y in zip(x_list, y_list):
                dis += (x - y) ** 2
            return math.sqrt(dis)

        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gt = [0 for _ in range(len(full_score))]
            gt[full_score.index(max(full_score[1:]))] = 1
            assert gt[0] != 1
            result.append(getdis(gt, full_score))  # +math.sqrt(2)*(1-full_score[0]))
        return result

    def Cos(self):

        def get_cos(X, Y):
            X = np.array(X)
            Y = np.array(Y)
            return X.dot(Y) / (np.sqrt(X.dot(X)) * np.sqrt(Y.dot(Y)))

        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gt = [0 for _ in range(len(full_score))]
            gt[full_score.index(max(full_score[1:]))] = 1
            assert gt[0] != 1
            result.append(-get_cos(full_score, gt))

        return result

    def Dis_p(self,datatype):
        eva = Evaluation(datatype)
        result = [[1 - x['score'], eva.get_iou([x['dis_score'][4:]], [x['dis_score'][0:4]])[0][0].item()] for x in
                  self.pre_list]
        res = np.array(result)
        # print(res.shape)

        # cm2 = plt.cm.get_cmap('RdYlBu_r')
        # loss = eva.get_loss(pre_list=self.pre_list)
        # Y = [1 - x['score'] for x in self.pre_list]
        # X = [eva.get_iou([x['dis_score'][4:]], [x['dis_score'][0:4]])[0][0].item() for x in self.pre_list]

        # zipped = zip(X, Y, loss)
        # sort_zipped = sorted(zipped, key=lambda x: (x[2], x[1], x[0]))
        # srt_ = zip(*sort_zipped)
        # srtd_X, srtd_Y, srtd_loss = [list(x) for x in srt_]
        # srtd_X = srtd_X[::-1]
        # srtd_Y = srtd_Y[::-1]
        # srtd_loss = srtd_loss[::-1]
        # for i in range(1, 100):
        #     rat = i / 100.
        #     Y.append(np.mean(res[np.where((res[:, 1] > rat) & (res[:, 1] < (rat + 0.01)))][:, 0]))
        #     X.append(rat)

        # plt.scatter(X, Y, c=loss, cmap=cm2)
        # plt.xlabel('IoU with rec')
        # plt.ylabel('Uncertainty')
        # plt.colorbar()
        # plt.legend()
        # plt.show()

        # min_max_scaler = preprocessing.MinMaxScaler()
        # result = min_max_scaler.fit_transform(result)
        return [x[0] * x[1] for x in result]

    def Cos_p0(self):

        def get_cos(X, Y):
            X = np.array(X)
            Y = np.array(Y)
            return X.dot(Y) / (np.sqrt(X.dot(X)) * np.sqrt(Y.dot(Y)))

        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gt = [0 for _ in range(len(full_score))]
            gt[full_score.index(max(full_score[1:]))] = 1
            assert gt[0] != 1
            result.append(-get_cos(full_score, gt) + full_score[0])

        return result

    def Dis_Cos(self):

        def get_cos(X, Y):
            X = np.array(X)
            Y = np.array(Y)
            return X.dot(Y) / (np.sqrt(X.dot(X)) * np.sqrt(Y.dot(Y)))

        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gt = [0 for _ in range(len(full_score))]
            gt[full_score.index(max(full_score[1:]))] = 1
            assert gt[0] != 1
            result.append([x['dis_score'], get_cos(full_score, gt)])

        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(result)
        return [x[0] - x[1] for x in result]

    def Gini(self):
        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            gini = 1 - sum([_ ** 2 for _ in full_score])
            result.append(gini)
        return result

    def my_entropy(self):
        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            entropy = sum([-p * math.log(p) for p in full_score])
            result.append([x['dis_score'], entropy])
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(result)
        return [x[0] + x[1] for x in result]

    def entropy_instance(self):
        result = []
        for i, x in enumerate(self.pre_list):
            full_score = x['full_score']
            result.append(sum([-p * math.log(p) for p in full_score]))

        return result

    def one_minus_pmax(self, part="all"):
        result = [(1 - x['score']) for x in self.pre_list]
        # for i, x in enumerate(self.pre_list):
        #     full_score = x['full_score']
        #     if part == "cls":
        #         result.append(1 - max(full_score[1:]))
        #     elif part == "all":
        #         result.append(1 - max(full_score))
        return result

    def entropy_image(self, T):
        def entropy_(x):
            if x[0] < T:
                return 0
            p_instn = x[0]
            return - p_instn * (math.log(p_instn))  # - (1 - p_instn) * (math.log(1 - p_instn))

        img_etps = [sum([entropy_(p) for p in img['two_score']]) for img in self.imgpre_list]
        img_id = [img['image_id'] for img in self.imgpre_list]
        zipped = zip(img_etps, img_id)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        srt_ = zip(*sort_zipped)
        _, srtd_img_id = [list(x) for x in srt_]
        result = []
        for i, x in enumerate(self.pre_list):
            result.append(srtd_img_id.index(x['image_id']))
        return result

    def random_instance(self):
        result = [i for i in range(len(self.pre_list))]
        random.shuffle(result)
        return result

    def random_image(self):
        result = []
        img_id = [img['image_id'] for img in self.imgpre_list]
        random.shuffle(img_id)
        for i, x in enumerate(self.pre_list):
            result.append(img_id.index(x['image_id']))
        return result

    def margin(self):
        eva = Evaluation()
        iou_list = eva.get_dif(pre_list=self.pre_list, dif_list=self.dif_list)
        print(len([x for x in iou_list if x <= 0.3]) / len(iou_list))

        # srt_iou_list = sorted(iou_list)
        # max_gap = -1
        # T_iou = 0
        # cm2 = plt.cm.get_cmap('RdYlBu_r')
        # eva2 = Evaluation()
        # loss = eva2.get_loss(pre_list=self.pre_list)
        # plt.scatter([1 - iou_list[i] for i in range(len(self.pre_list))],
        #             [1 - self.pre_list[i]['score'] for i in range(len(self.pre_list))], c=loss, cmap=cm2)
        # plt.xlabel('Consistance')
        # plt.ylabel('Uncertainty')
        # plt.colorbar()
        # plt.legend()
        # plt.show()
        def one_two_(x):
            return - (1.7 * x[-1] + 1.3 * x[-2]) ** 2

        result = [[(1 - iou_list[i]), one_two_(sorted(self.pre_list[i]['full_score'][1:]))] for i in
                  range(len(self.pre_list))]
        # min_max_scaler = preprocessing.MinMaxScaler()
        # result = min_max_scaler.fit_transform(result)
        ## if iou_list[i] ==0 else (1 - self.pre_list[i]['score']) for
        return [x[1] for x in result]

    def difference(self,datatype):
        eva = Evaluation(datatype)
        iou_list = eva.get_dif(pre_list=self.pre_list, dif_list=self.dif_list)
        print(len([x for x in iou_list if x <= 0.3]) / len(iou_list))

        # srt_iou_list = sorted(iou_list)
        # max_gap = -1
        # T_iou = 0
        # cm2 = plt.cm.get_cmap('RdYlBu_r')
        # eva2 = Evaluation()
        # loss = eva2.get_loss(pre_list=self.pre_list)
        # plt.scatter([1 - iou_list[i] for i in range(len(self.pre_list))],
        #             [1 - self.pre_list[i]['score'] for i in range(len(self.pre_list))], c=loss, cmap=cm2)
        # plt.xlabel('Consistance')
        # plt.ylabel('Uncertainty')
        # plt.colorbar()
        # plt.legend()
        # plt.show()

        result = [[(1 - iou_list[i]), (1 - self.pre_list[i]['score'])] for i in
                  range(len(self.pre_list))]
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(result)
        ## if iou_list[i] ==0 else (1 - self.pre_list[i]['score']) for
        return [math.fabs(x[0] - 0.5) * x[1] for x in result]

    def difference_1(self):
        eva = Evaluation()
        iou_list = eva.get_dif(pre_list=self.pre_list, dif_list=self.dif_list)
        print(len([x for x in iou_list if x <= 0.3]) / len(iou_list))

        result = [[(1 - iou_list[i]), (1 - self.pre_list[i]['score'])] for i in
                  range(len(self.pre_list))]
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(result)
        return [(x[0] - 0.5) * x[1] for x in result]

    def difference_2(self):
        eva = Evaluation()
        iou_list = eva.get_dif(pre_list=self.pre_list, dif_list=self.dif_list)
        print(len([x for x in iou_list if x <= 0.3]) / len(iou_list))

        result = [[(1 - iou_list[i]), (1 - self.pre_list[i]['score'])] for i in
                  range(len(self.pre_list))]
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(result)
        return [(0.5 - x[0]) * x[1] for x in result]

    def one_vs_two(self, T):
        def one_two_(x):
            if x[0] < T:
                return 0
            return (1 - (x[0] - x[1])) ** 2

        img_one_two = [sum([one_two_(p) for p in img['two_score']]) for img in self.imgpre_list]
        img_id = [img['image_id'] for img in self.imgpre_list]
        zipped = zip(img_one_two, img_id)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        srt_ = zip(*sort_zipped)
        _, srtd_img_id = [list(x) for x in srt_]
        result = []
        for i, x in enumerate(self.pre_list):
            result.append(srtd_img_id.index(x['image_id']))
        return result

    def gini(self, T):
        def gini_(x):
            if x[0] < T:
                return 0
            return x[0] ** 2

        img_one_two = [1 - sum([gini_(p) for p in img['two_score']]) for img in self.imgpre_list]
        img_id = [img['image_id'] for img in self.imgpre_list]
        zipped = zip(img_one_two, img_id)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        srt_ = zip(*sort_zipped)
        _, srtd_img_id = [list(x) for x in srt_]
        result = []
        for i, x in enumerate(self.pre_list):
            result.append(srtd_img_id.index(x['image_id']))
        return result
