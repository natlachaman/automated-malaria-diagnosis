import pickle
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
# sns.set_style("whitegrid")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    sns.set_style("whitegrid", {'axes.grid': False})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure()
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(cm, linewidths=.5, cmap=cmap, fmt=fmt, annot=True)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes)) + 0.5
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks[::-1], classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.
    Precision is modified to ensure that it does not decrease as recall
    decrease.
    Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
    Raises:
    ValueError: if the input is not of the correct format
    Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
    """
    if precision is None:
        if recall is not None:
          raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(recall, np.ndarray):
        raise ValueError("precision and recall must be numpy array")

    # if precision.dtype != np.float or recall.dtype != np.float:
    #     raise ValueError("input must be float numpy array.")

    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")

    if not precision.size:
        return 0.0

    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")

    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")

    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # # interpolation for fixes recall values
    # r_levels = list(np.linspace(0.1, 1, 10))
    # precision_at_k = np.zeros((11,))
    # precision_at_k[0] = 1.0
    # for i, level in enumerate(r_levels):
    #     k = min(recall, key=lambda x: abs(x - level))
    #     k_idx = max(np.where(recall == k)[0])
    #     precision_at_k[i+1] = np.mean(precision[:k_idx])

    # Preprocess precision to be a non-decreasing array
    # precision = np.concatenate([[0], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # indices = np.where(recall[1:] != recall[:-1])[0] + 1
    # average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    # average_precision = np.sum(np.diff(recall) * precision[1:])
    average_precision = np.mean(precision)

    # return average_precision, precision_at_k, np.asarray([0.0] + r_levels)
    return average_precision, precision, recall

# 256 'model_2018_3_29_9_23_39' 'model.118-1.13.json'
# 512 'model_2018_1_29_8_44' 'model.119-1.25.json'
# 768 'model_2018_3_26_10_55_34' 'model.111-1.44.json'
# transfer 'model_2018_3_15_20_37_39' 'model.114-2.04.json'

# 256 2 'model_2018_5_16_10_58_23' 'model.118-1.15.json'
# 256 3 'model_2018_5_21_20_12_25_83' 'model.117-2.63.json'
# 256 4 'model_2018_5_16_10_59_16_1' 'model.119-2.65.json'
# 512 5 'model_2018_5_22_6_6_10_25' 'model.119-2.02.json' (scracth)
# 512 5 'model_2018_5_15_20_39_23_2' 'model.119-1.98.json'

saveto = [
        './results/model_2018_3_29_9_23_39/results/',
        './results/model_2018_1_29_8_44/results/',
        './results/model_2018_3_26_10_55_34/results/'
    ]

# saveto = [
    # './results/model_2018_3_29_9_23_39/results/',
    # './results/model_2018_1_29_8_44/results/',
    # './results/model_2018_5_16_10_58_23/results/',
    # './results/model_2018_5_21_20_12_25_83/results/',
    # './results/model_2018_5_16_10_59_16_1/results/',
    # './results/model_2018_5_22_6_6_10_25/results/',
    # './results/model_2018_5_15_20_39_23_2/results/'
# ]

# colors = ['aquamarine', 'coral', 'mediumvioletred', 'teal', 'royalblue', 'khaki', 'lime']
colors = ['aquamarine', 'coral', 'royalblue', 'teal']
styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']
# styles = ['solid', 'dashed']

from scipy.interpolate import interp1d, UnivariateSpline
cm = np.zeros((3, 3))
tags = {1: 'Pf', 2: 'WBC'}
fig2, ax2 = plt.subplots()
for c in [1, 2]:
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    # fig3, ax3 = plt.subplots()
    # fig4, ax4 = plt.subplots()
    # ylim, xlim = 0, 0

    # for m, model in enumerate(['SSD-256', 'SSD-512', 'SSD-768']):
    for m, model in enumerate(['SSD-256',
                               'SSD-512',
                               'SSD-768'
                               # 't-block2-SSD-256',
                               # 't-block3-SSD-256',
                               # 't-block4-SSD-256*',
                               # 't-block5-SSD-512',
                               # 't-block5-SSD-512',

                               ]):
        f_errors = saveto[m] + 'I-III_ErrTypes_NMS_0.4.pickle'

        with open(f_errors, 'rb') as f:
            dict_of_err, thresholds = pickle.load(f)

        cumsumTPs, cumsumFPs, cumsumFNs = [], [], []
        for t in tqdm(thresholds[::-1]):
            # TP, FP, FN, gt = 0, 0, 0, 0
            cm = np.zeros((3, 3))

            for k in dict_of_err:
                if len(dict_of_err[k]) > 0:
                    # TP += dict_of_err[k][t][c]['tp']
                    # FP += dict_of_err[k][t][c]['fp']
                    # FN += dict_of_err[k][t][c]['fn']
                    # gt += dict_of_err[k][t][c]['tp'] + dict_of_err[k][t][c]['fn']
                    cm += dict_of_err[k][t]['cm']
            # cm /= (len(dict_of_err) -1)
            TP = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
            FP = cm[0][1] + cm[0][2]
            FN = cm[1][0] + cm[2][0]
            gt = TP + FN
            cumsumTPs.append(TP)
            cumsumFPs.append(FP)
            cumsumFNs.append(FN)

        fps = np.asarray(cumsumFPs) / float(len(list(dict_of_err.keys())))
        fns = np.asarray(cumsumFNs) / np.float32(np.asarray(cumsumTPs) + np.asarray(cumsumFNs))
        recall = np.asarray(cumsumTPs) / float(gt)
        precision = np.nan_to_num(np.asarray(cumsumTPs) / np.float32(np.asarray(cumsumTPs) + np.asarray(cumsumFPs)))
        precision = np.clip(precision, a_min=0, a_max=1)
        ap, precision, recall = compute_average_precision(precision, recall)

        # print(model, c, gt, ap)
        # # Error Detection Curve per class
        # ax1.step(fps, fns,  color=colors[m], linestyle=styles[c-1], label=model)
        # ax1.set_xlabel('False Positives per Image')
        # ax1.set_ylabel('Miss rate')
        # ax1.set_xlim([0, 0.5])
        # ax1.set_ylim([0.0, 1.0])
        # ax1.legend(loc='lower left')

        # FROC curve per class (manually)
        fps = np.concatenate([[0], fps, [8]])
        # f = interp1d(recall, fps)
        # f = UnivariateSpline(recall, fps, k=3)
        ynew = np.linspace(0.0, 1.0, 100, endpoint=True)

        ax2.plot(fps, recall, color=colors[m], linestyle=styles[c - 1], label=model)
        # ax2.plot(f(ynew), ynew, color=colors[m], linestyle=styles[c - 1], label=model)
        ax2.set_xlabel('False Positives per Image')
        ax2.set_ylabel('Detection rate')
        # ylim = ylim if recall.max() > ylim > 0 else recall.max()
        # ylim = 0.8
        # ax2.set_ylim([0.0, 1.0])
        # xlim = xlim if fps.max() > xlim > 0 else fps.max()
        ax2.set_xlim([0.0, 5])
        # ax2.legend(loc='lower right')

#----------------------------------------------------------------------------------------------------------------#
#       tensorflow style P-R & FROC curve
# ---------------------------------------------------------------------------------------------------------------#
#         recall = recall[:-1]
#         precision = precision[:-1]
#         optimal_idx = np.argmin(recall - precision)
#         cut_off = thresholds[np.argmin(np.abs(thresholds - 0.5))]
#
#         print('model', model)
#         print('class', c)
#         # print('optimal threshold', cut_off)
#         # print('optimal recall', recall[optimal_idx])
#         print('recall@AP', np.where(np.asarray(precision) - ap < 0.001, recall, np.zeros_like(recall)).max())
#         print('ap', ap)

        # # precision-recall curve per class (tf style)
        # points = zip(recall, precision)
        # points = sorted(points, key=lambda point: point[0])
        # x1, y1 = zip(*points)
        #
        # f = UnivariateSpline(recall, precision, k=3)
        # # f = interp1d(recall, precision)
        # xnew = np.linspace(min(recall), max(recall), 100)
        #
        # # ax3.plot(recall, precision, color=colors[m], linestyle=styles[m], label='{0}, AP={1:0.4f}'.format(model, ap))
        # plt.step(xnew, f(xnew), color=colors[m], linestyle=styles[m], label='{0}, AP={1:0.4f}'.format(model, ap))
        # ax3.set_xlabel('Recall')
        # ax3.set_ylabel('Precision')
        # # ax3.set_ylim([0.0, 1.0])
        # # xlim = xlim if recall.max() > xlim > 0 else recall.max()
        # # xlim = 0.78 if c == 1 else 0.95
        # # ax3.set_xlim([0.0, xlim])
        # # ylim = (precision.min(), precision.max()) if c == 1 else (0.00, 0.55)
        # # ax3.set_ylim(ylim)
        # # bbox_to_anchor = (0.45, 0.3) if c == 1 else (0.6, 0.4)
        # # ax3.legend(bbox_to_anchor=bbox_to_anchor)
        # ax3.legend(loc='lower left')

        # # make confusion matrix, optimized per class
        # for k in dict_of_err:
        #     cm += dict_of_err[k][cut_off]['cm']
        # cm /= len(dict_of_err)
        # fig5 = plot_confusion_matrix(cm, ['BG', 'PF', 'WBC'], cmap='PuBu', normalize=True)
        # fig5.savefig(os.path.join('./', 'confusion-matrix-{}_{}.png'.format(model, c)))
        # # plt.show()
        # # 'PuBuGn'

    # fig1.savefig(os.path.join('./', 'fp-fnr-curve-{}.png'.format(tags[c])))
    # fig1.show()
    # fig2.savefig(os.path.join('./', 'froc-curve-{}.png'.format(tags[c])))
    # fig2.show()
    # fig3.savefig(os.path.join('./', 't_precision-recall-{}.png'.format(tags[c])))
    # fig3.show()
    # fig4.savefig(os.path.join('./', 'froc-curve-cumsum-{}.png'.format(tags[c])))
    # fig4.show()
    break

# # In total 3x3 lines have been plotted
lines = ax2.get_lines()
legend1 = plt.legend([lines[i] for i in [0, 1, 2]], ['SSD-256', 'SSD-512', 'SSD-768'], loc=4)
ax2.add_artist(legend1)
# bbox_to_anchor = (0.55, 0.17) #(0.6, 0.4)
# legend2 = plt.legend([lines[i] for i in [0, 4]], ['Pf', 'WBC'], bbox_to_anchor=bbox_to_anchor)
# ax2.add_artist(legend2)

# leg = ax2.get_legend()
# for i in range(2):
#     leg.legendHandles[i].set_color('black')

fig2.savefig(os.path.join('./', 'froc-curve-detection.png'))
fig2.show()
