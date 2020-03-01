# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from utils.metrics import nms_for_single_image
from utils.bboxes import np_bboxes_sort
import seaborn as sns


# plt.style.use('fivethirtyeight')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# # plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12

# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes_original(img, classes, bboxes, saveto, figsize=(10,10), linewidth=1.5):
    '''
        Plot bounding boxes on a given image. No batch support.

    :param img: image array (height, width, 3)
    :param classes: numpy array (num_classes, )
    :param bboxes: dictionary {class (int): box coordinates (?, 4) }
    :param saveto: file to save to including path (string)
    :param figsize: figure size (tuple of ints)
    :param linewidth: width of box sides (float)
    '''
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    # handles = []x1
    for i in classes:
        cls_id = int(i)
        colors[cls_id] = (random.random(), random.random(), random.random())
        if bboxes[i].any():
            for j in range(len(bboxes[i])):
                ymin = int(bboxes[i][j, 0] * height)
                xmin = int(bboxes[i][j, 1] * width)
                ymax = int(bboxes[i][j, 2] * height)
                xmax = int(bboxes[i][j, 3] * width)

                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     label=None,
                                     edgecolor=colors[cls_id],
                                     linewidth=linewidth)

                plt.gca().text(xmin, ymin - 2,
                               str(i),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=12, color='white')

                plt.gca().add_patch(rect)
            # h, = plt.plot(0., 0., label=str(i), color=colors[cls_id])
            # handles.append(h)
            # plt.legend(handles=handles, labels=[str(i) for i in classes])
    print('saved to: {}'.format(saveto))
    fig.savefig(saveto)

# =========================================================================== #
# Plot utilities for training...
# =========================================================================== #
def plot_anchors_on_an_image(image, ssd_net, dir, name):
    """
        Plots anchor boxes on an image.

    :param image: (numpy array) of pixels
    :param ssd_net: (obj) SSDNet class object
    :param dir: saveto path
    :param name: file name to save to

    :return: None
    """
    anchors = ssd_net.anchors()
    boxes = []

    # batch = 1
    layers = np.arange(len(anchors), dtype='int32')
    for layer in layers:
        x, y, w, h = anchors[layer]
        x, y, w, h = np.reshape(x, -1), np.reshape(y, -1), np.reshape(w, -1), np.reshape(h, -1)

        Y, H = np.meshgrid(y, h)
        X, W = np.meshgrid(x, w)
        ymin, ymax = Y.ravel() - (H.ravel() / 2.), Y.ravel() + (H.ravel() / 2.)
        xmin, xmax = X.ravel() - (W.ravel() / 2.), X.ravel() + (W.ravel() / 2.)
        p = np.random.permutation(len(ymin))[:10]
        # p = np.arange(len(ymin))[10000:10010]
        print('{} anchors in layer {}'.format(len(ymin), layer))
        boxes.append(np.concatenate([ymin[p, None], xmin[p, None], ymax[p, None], xmax[p, None]],
                                    axis=1))
    plt_bboxes_original(image, layers, boxes, os.path.join(dir, name),
                        figsize=(10, 10), linewidth=1.5)


from collections import OrderedDict
def plot_boxes_per_layer_on_an_image(b_image, b_classes, b_scores, b_bboxes, layer_names, dir, name):
    colors_tableau = [
          # (255, 255, 255),
          # (31, 119, 180),
          # (174, 199, 232),
          # (255, 127, 14),
          (255, 187, 120),
          # (44, 160, 44),
          # (152, 223, 138),
          (214, 39, 40),
          # (255, 152, 150),
          # (148, 103, 189),
          (197, 176, 213),
          # (140, 86, 75),
          # (196, 156, 148),
          (227, 119, 194),
          # (247, 182, 210),
          # (127, 127, 127),
          (199, 199, 199),
          # (188, 189, 34),
          # (219, 219, 141),
          (23, 190, 207),
          # (158, 218, 229)
          (31, 119, 180)
    ]

    for n, (image, l_classes, l_scores, l_bboxes) in enumerate(zip(b_image, b_classes, b_scores, b_bboxes)):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow((image + 1) / 2)
        for i, (classes, scores, bboxes) in enumerate(zip(l_classes, l_scores, l_bboxes)):
            height, width = image.shape[:-1]

            # sort decreasing order of confidence values on the same layer
            # idxes = np.argsort(-scores)
            # classes = classes[idxes][:50]
            # scores = scores[idxes][:50]
            # bboxes = bboxes[idxes][:50]

            idxes = np.where(scores > 0.5)[0]
            classes = classes[idxes]
            # scores = scores[idxes]
            bboxes = bboxes[idxes]

            if bboxes.any():
                bboxes2 = []
                for ymin, xmin, ymax, xmax in bboxes:
                    if xmax - xmin == 0:
                        xmin -= (25 / width)
                        xmax += (25 / width)
                    if ymax - ymin == 0:
                        ymin -= (25 / height)
                        ymax += (25 / height)
                    bboxes2.append((ymin, xmin, ymax, xmax))
                bboxes = np.array(bboxes2)

            # supress redundant detections on a same layer
            bboxes, idxes = nms_for_single_image(bboxes, 0.1)

            # plot remaining boxes
            if bboxes.any():
                classes = classes[idxes]

                for cls, (ymin, xmin, ymax, xmax) in zip(classes, bboxes):
                    ymin = int(ymin * height)
                    xmin = int(xmin * width)
                    ymax = int(ymax * height)
                    xmax = int(xmax * width)

                    # print(xmax-xmin)
                    # print(ymax-ymin)

                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin,
                                         fill=False,
                                         label=layer_names[i],
                                         edgecolor=np.array(colors_tableau[i]) / 255.,
                                         linewidth=1.5)

                    plt.gca().text(xmin, ymin - 2,
                                   cls, bbox=dict(facecolor=np.array(colors_tableau[i]) / 255., alpha=0.5),
                                   fontsize=12, color='white')

                    plt.gca().add_patch(rect)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        print('saved to: {}'.format(os.path.join(dir, name + '_' + str(n))))
        fig.savefig(os.path.join(dir, name + '_' + str(n)))


def plot_performance(history, saveto):
    """
        Plots training performance.

    :param history: (obj) history object returned by fit_generator in Keras
    :param saveto: path to save plots to

    :return: None
    """
    fig, ax = plt.subplots(1, 1)
    epochs = np.arange(len(history['loss']), dtype='int32')
    ax.plot(epochs, history['loss'], 'r', label='train')
    ax.plot(epochs, history['val_loss'], 'b', label='val')
    # ax.set_title('Total Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 3])
    ax.legend()
    fig.savefig(os.path.join(saveto, 'total_loss.png'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, history['loc_loss'], 'r', label='train')
    ax.plot(epochs, history['val_loc_loss'], 'b', label='val')
    # ax.set_title('Localisation Loss (Smooth L1-norm)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 3])
    ax.legend()
    fig.savefig(os.path.join(saveto, 'loc_loss.png'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, history['conf_loss'], 'r', label='train')
    ax.plot(epochs, history['val_conf_loss'], 'b', label='val')
    # ax.set_title('Confidence Loss (cross entropy)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 3])
    ax.legend()
    fig.savefig(os.path.join(saveto, 'conf_loss.png'))


def plt_bboxes(img, cls, scores, bboxes, colors, ax, linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """

    ax.imshow(img)
    sns.set_style("whitegrid", {'axes.grid': False})
    height, width, _ = img.shape

    for i, bbox in enumerate(bboxes):
        ymin = int(bbox[0] * height)
        xmin = int(bbox[1] * width)
        ymax = int(bbox[2] * height)
        xmax = int(bbox[3] * width)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls],
                             linewidth=linewidth)
        plt.gca().add_patch(rect)
        if type(scores[i]) is not str:
            plt.gca().text(xmin, ymin - 2,
                           '{0:s} | {1:.2f}'.format(str(cls), scores[i]),
                           bbox=dict(facecolor=colors[cls], alpha=0.5),
                           fontsize=12, color='white')
        else:
            print(scores[i])
            plt.gca().text(xmin, ymin - 2,
                           '{0:s} | {1:s}'.format(str(cls), str(scores[i])),
                           bbox=dict(facecolor=colors[cls], alpha=0.5),
                           fontsize=12, color='white')

    plt.axis('off')


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