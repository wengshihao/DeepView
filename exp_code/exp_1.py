import json
import matplotlib.pyplot as plt
from Metrics import Metrics
from evaluation import Evaluation
from visualization import visualization
from RQ3engine import RQ3engine


def RUN_RQ_12(score_throd, modeltype, datatype, runtype):
    with open('./data/' + runtype + modeltype + '_' + datatype + score_throd + '.json') as f1:
        pre_list = json.load(f1)
    # with open('./data/' + runtype + 'gaus_' + modeltype + '_' + datatype + score_throd + '.json') as f1:
    #     dif_list = json.load(f1)
    with open('./data/' + runtype + modeltype + '_' + datatype + 'image.json') as f2:
        imgpre_list = json.load(f2)
    lamda = 1

    # img_title = '$cls\ loss\ +\ reg\ loss\ (normalized) with threshold '+score_throd+'$'
    img_title = 'metric evalution'
    eva = Evaluation(datatype=datatype, runtype=runtype)
    eva2 = Evaluation(datatype=datatype, runtype=runtype)
    eva3 = Evaluation(datatype=datatype, runtype=runtype)
    fault_list, fault_type_num = eva.get_faulttype(pre_list)
    pre_loss = eva2.get_loss(pre_list)
    _, _, tp_list = eva3.get_score_iou(pre_list)
    instance_num = len(pre_loss)

    result = {}
    metrics = Metrics(pre_list=pre_list, dif_list=None, imgpre_list=imgpre_list)
    # result['ours(difference)'] = metrics.difference(datatype)
    result['DeepView'] = metrics.Dis_p(datatype)
    # result['$1-p_{max}$(instance)'] = metrics.one_minus_pmax(part="all")
    result['1vs2-Sum'] = metrics.one_vs_two(0)
    result['Entropy'] = metrics.entropy_image(0)
    result['DeepGini'] = metrics.gini(0)
    result['Random-Instance'] = metrics.random_instance()
    result['Random-Image'] = metrics.random_image()
    metric_names = [_[0] for _ in result.items()]
    colorlist=['xkcd:red','xkcd:peach','xkcd:green','xkcd:light purple','xkcd:black','xkcd:grey']
    percent_throd = 0.1
    vis = visualization()
    X = []
    Tro_diversity = []
    x_list = []
    Tro_effective = []
    # fig = plt.figure(figsize=(18, 10), dpi=100)
    # effective = fig.add_subplot(2, 2, 1)
    # diversity = fig.add_subplot(2, 2, 2)
    # effective = plt.figure(1)
    # rauctable = fig.add_subplot(2, 1, 2)
    # plt.title(runtype + modeltype + ' on ' + datatype + ' with T = ' + str(int(score_throd) / 100), fontsize=18,
    #           fontweight='bold', bbox=dict(edgecolor='blue', alpha=0.65))
    # effective.set_title('effective')
    # diversity.set_title('diversity')
    row = []
    val = []
    for i in range(len(result)):
        metric = result[metric_names[i]]
        zipped = zip(metric, pre_loss, fault_list, tp_list, pre_list)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        srt_ = zip(*sort_zipped)
        srtd_metric_list, srtd_loss_list, srtd_fault_list, srtd_tp_list, srtd_pre_list = [list(x) for x in srt_]
        # rq3 = RQ3engine(slice=[0.1])
        # rq3.map2image(srtd_pre_list[::-1], metric_names[i], modeltype)
        X, Y, Tro_diversity = vis.plt_fault_type_rauc(srtd_fault_list[::-1], fault_type_num)
        plt.plot(X, Y, label=metric_names[i],color=colorlist[i])
        x_list, Tro_effective, y_list, rauc_1, rauc_2, rauc_3, rauc_5, rauc_all = eva.RAUC_cls(
            metric_tplist=srtd_tp_list[::-1])
        print(metric_names[i]+'div:'+' '+str(sum(Y)/sum(Tro_diversity)*100))
        # plt.plot(x_list, y_list, label=metric_names[i])
        val.append([round(rauc_1 * 100, 2), round(rauc_2 * 100, 2), round(rauc_3 * 100, 2), round(rauc_5 * 100, 2),
                    round(rauc_all * 100, 2)])
        # print(metric_names[
        #           i] + ' RAUC-100:{:.3f}, RAUC-200:{:.3f}, RAUC-300:{:.3f}, RAUC-500:{:.3f}, RAUC-all:{:.3f},'.format(
        #     rauc_1, rauc_2, rauc_3, rauc_5, rauc_all))
        row.append(metric_names[i])

    col = ['RAUC-500', 'RAUC-1000', 'RAUC-2000', 'RAUC-5000', 'RAUC-all']
    # the_table = rauctable.table(cellText=val, colLabels=col, rowLabels=row, loc='center', cellLoc='center',
    #                             rowLoc='center')
    # the_table.scale(1, 1.5)
    # the_table.set_fontsize(14)
    # rauctable.axis('off')
    plt.plot(X, Tro_diversity, label='Theoretical',color='tab:blue')

    # plt.plot(x_list, Tro_effective, label='Tro')
    plt.xlabel('$Number\ of\ prioritized\ test\ instances$')
    plt.ylabel('$Number\ of\ error\ type\ detected$')
    #
    # diversity.legend()
    # effective.legend()
    plt.legend()
    # plt.show()
    plt.savefig(
        'C:/Users/WSHdeWindows/Desktop/picture/RQ2new/RQ2' + modeltype + '_' + datatype + '_' + score_throd + '.pdf')
    # plt.savefig(
    #     './pictures/' + runtype + modeltype + '_' + datatype + '_' + 'T=' + str(int(score_throd) / 100) + '.png',
    #     bbox_inches='tight', dpi=800)
    plt.show()

if __name__ == "__main__":
    RUN_RQ_12(score_throd='050', modeltype='SSD', datatype='VOC', runtype='')
