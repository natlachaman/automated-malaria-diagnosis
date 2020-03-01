import os, pickle, sys
sys.path.append('../')

import argparse
from itertools import chain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'

import utils
from utils.metrics import MultiboxLoss, match_truth_and_predictions, nms_for_single_image, compute_average_precision
from utils.visualization import plot_performance, plt_bboxes, plot_confusion_matrix
from utils.model import to_relative_coordinates
from network import SSDNet
from sklearn.metrics import confusion_matrix
from scipy.interpolate import UnivariateSpline


import global_variables


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    source: (https://nolanbconaway.github.io/blog/2017/softmax-numpy)

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def np_detect_objects(model, ssd_net, images, args):
    '''
            Uses the model to detect class objects on a set of sub-images.
        :param model: (object) Model instance from Keras
        :param ssd_net: (object )SSDNet class object
        :param images: (list) of test images to predict on
        :param args: (namedtuple) evaluation arguments relevant to the script

        :return: classes, (list of numpy arrays) where every row represents an object class for an object in an image.
                 scores, (list of numpy arrays) where every row represents conf scores for an object in an image.
                 bboxes, (list of numpy arrays) where every row represents bboxes coordinates for an object in an image.
        '''
    # get output from network
    logits, localisations = model.predict(np.array(images), batch_size=args.batch_size, verbose=2)
    predictions = softmax(logits, axis=-1)

    # detect objects
    classes, scores, bboxes = ssd_net.bboxes_detect_numpy([predictions, localisations], apply_nms=False)

    return classes, scores, bboxes


def draw_predicted_bboxes_on_images(images, classes, scores, bboxes, args):
    """
        Draw color-coded bounding boxes around identified class objects.
    :param images: (list) of images as numpy arrays
    :param classes: (list of numpy arrays) where every row represents an object class for an object in an image.
    :param scores: (list of numpy arrays) where every row represents conf scores for an object in an image.
    :param bboxes: (list of numpy arrays) where every row represents bboxes coordinates for an object in an image.
    :param args: general parameters for the evaluation script.

    :return: None
    """

    # set colors per class once
    colors = dict()
    class_id = np.arange(1, args.ssd_params.num_classes)
    for cls in class_id:
        colors[cls] = (np.random.random(), np.random.random(), np.random.random())

    # loop over a batch of images
    class_id = np.arange(1, args.ssd_params.num_classes)
    for i, img in enumerate(images):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # loop over each object class and plot it on image i
        for cls in class_id:
            select = np.where(classes[i] == cls)[0]
            plt_bboxes(img,
                       cls,
                       scores[i][select],
                       bboxes[i][select],
                       colors,
                       ax,
                       linewidth=1.5)

        # plt.show()
        print('saved to: {}'.format(os.path.join(args.saveto, 'image_' + str(i))))
        fig.savefig(os.path.join(args.saveto, 'image_' + str(i)))


def draw_predicted_and_target_bboxes_on_images(images, tandp, args):
    '''
        Draw color-coded bounding boxes around identified class objects.
    :param images:
    :param classes:
    :param scores:
    :param bboxes:
    :param args:

    :return:
    '''
    colors = dict()
    class_id = np.arange(1, args.ssd_params.num_classes)

    # loop over a batch of images
    for i, img in tqdm(enumerate(images), desc='[INFO] Drawing objects on images...'):
        img = ((img + 1) / 2)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # loop over set of true and pred annotations
        for k, v in global_variables.COLORS.items():
            classes, scores, bboxes = tandp[k]

            # plot one class at a time (per image)
            for cls in class_id:
                colors[cls] = v
                select = np.where(classes[i] == cls)[0]
                if len(select) > 0:

                    if k is 'pred':
                        plt_bboxes(img,
                                   cls,
                                   scores[i][select],
                                   bboxes[i][select],
                                   colors,
                                   ax,
                                   linewidth=1.5)

                    if k is 'true':
                        plt_bboxes(img,
                                   cls,
                                   ['    ']*len(bboxes[i]),
                                   bboxes[i][select],
                                   colors,
                                   ax,
                                   linewidth=2.)

        # plt.show()
        f = os.path.join(args.saveto, 'test_image_' + str(i))
        print('saved to: {}'.format(f))
        try:
            fig.savefig(f, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        except ValueError:
            print('[WARNING]: predicted boxes are way too out of image frame.')
            continue


def tf_detect_objects(model, images, args, ssd_net, sess):
    '''
        Uses the model to detect class objects (tf).
    :param model:
    :param images:
    :param args:
    :param ssd_net:
    :param sess:
    :return:
    '''

    # predict batch
    logits, localisations = model.predict(np.array(images), batch_size=args.batch_size, verbose=2)
    predictions = softmax(logits, axis=-1)

    # get detected boxes from batch
    tf_predictions = tf.placeholder(shape=logits.shape, dtype='float32')
    tf_localisations = tf.placeholder(shape=localisations.shape, dtype='float32')
    rscores, rbboxes = sess.run(ssd_net.bboxes_detect([tf_predictions, tf_localisations]),
                                feed_dict={tf_predictions: predictions,
                                           tf_localisations: localisations})

    return rscores, rbboxes


def define_model_from_config(args, fconfig):
    """
        Reads config file and extract value parameters of a particular model (the one to be evaluated).
        These params are necessary for the bboxes detection.

    :param args: (namedtuple) script arguments relevant to the evaluation process
    :param fconfig: (string) path to config file

    :return: None
    """
    with open(fconfig, 'r') as f:
        # check every line in the text file
        for line in f:

            # if the line is of the format variable:value, then is an model parameter
            if ':' in line:
                field, value = line.split(':')

                # check if the field exists in ssd_params
                if hasattr(args.ssd_params, field):

                    # convert to the right data types depending on the field:
                    if field in ['img_shape', 'feat_layers', 'anchor_size_bounds',
                                 'anchor_steps', 'normalizations', 'prior_scaling']:
                        for ch in ['[', ']', '\n', '(', ')', ' ', ]:
                            value = value.replace(ch, '')
                        _value = list()
                        for i in value.split(','):
                            if i.isdigit() or i.startswith('-'):
                                _value.append(int(i))
                            elif i.replace('.', '').isdigit():
                                _value.append(float(i))
                            else:
                                _value.append(i[1:-1])
                        setattr(args.ssd_params, field, _value)

                    if field in ['num_classes', 'no_annotation_label', 'anchor_offset', 'matching_threshold']:
                        for ch in ['\n', ' ']:
                            value = value.replace(ch, '')
                        if value.isdigit():
                            setattr(args.ssd_params, field, int(value))
                        else:
                            setattr(args.ssd_params, field, float(value))

                    if field in ['feat_shapes', 'anchor_sizes', 'anchor_ratios']:
                        for ch in ['[', ']', ' ', '\n']:
                            value = value.replace(ch, '')
                        if field == 'anchor_ratios':
                            value = '),'.join(filter(lambda x: x.isdigit(), value))
                        _value = list()
                        for j in value.split('),'):
                            for ch in ['(', ')', ' ']:
                                j = j.replace(ch, '')
                            _tuple = list()
                            for i in j.split(','):
                                if i.isdigit():
                                    _tuple.append(int(i))
                                elif i.replace('.', '', 1).isdigit():
                                    _tuple.append(float(i))
                                elif i is None or i == 'None':
                                    _tuple.append(None)
                            _value.append(tuple(_tuple))
                        setattr(args.ssd_params, field, _value)


def to_original_coordinates(image, classes, scores, bboxes, XY, args):
    '''
        Merge detections from cropped images and apply NMS.
    '''

    # offsets to relative coordinates
    XY = to_relative_coordinates(np.array(XY), image.shape)

    obboxes, oclasses, oscores = [], [], []
    for b, c, s, xy in zip(bboxes, classes, scores, XY):
        if b.any():
            ymin, xmin, ymax, xmax = b.T
            yoffset, xoffset, _, _ = xy

            # to pixels
            ymin *= args.ssd_params.img_shape[0]
            xmin *= args.ssd_params.img_shape[0]
            ymax *= args.ssd_params.img_shape[0]
            xmax *= args.ssd_params.img_shape[0]

            # (cx, cy, w, h)
            w, h = xmax - xmin, ymax - ymin
            cx, cy = xmin + (w / 2.), ymin + (h / 2.)

            # center coordinates + offset
            cx += (xoffset * image.shape[1])
            cy += (yoffset * image.shape[0])

            # back to (ymin, xmin, ymax, xmax)
            xmin, xmax = cx - (w / 2.), cx + (w / 2.)
            ymin, ymax = cy - (h / 2.), cy + (h / 2.)

            # some detections have no volume (something's up with the predictions)
            # mask = np.logical_and(xmax - xmin == 0, classes == 1)
            # xmax[mask] += 25
            # xmin[mask] -= 25
            # ymax[mask] += 25
            # ymin[mask] -= 25
            #
            # mask = np.logical_and(xmax - xmin == 0, classes == 2)
            # xmax[mask] += 50
            # xmin[mask] -= 50
            # ymax[mask] += 50
            # ymin[mask] -= 50

            # to relative
            nbb = to_relative_coordinates(np.hstack([ymin[:, None],
                                                     xmin[:, None],
                                                     ymax[:, None],
                                                     xmax[:, None]]),
                                          image.shape)

            obboxes.extend(nbb)
            oclasses.extend(c)
            oscores.extend(s)

    # apply NMS to bboxes over full image
    bboxes, idx = nms_for_single_image(obboxes, threshold=0.1)
    classes = np.array([oclasses[i] for i in idx])
    scores = np.array([oscores[i] for i in idx])

    return classes, scores, bboxes


def evaluate_function(args):
    """
        Evaluation pipeline.

    :param args: (namedtuple) script arguments relevant to the evaluation process

    :return: plots results with matplotlib
    """
    # create results folder
    if not os.path.exists(args.saveto):
        os.makedirs(args.saveto)

    # load data
    gt = pickle.load(open(args.gt, 'rb'), encoding='latin1')

    # define model parameters
    define_model_from_config(args, os.path.join(args.model_dir, 'model.config'))
    ssd_net = SSDNet(args)

    # load trained weights
    losses = MultiboxLoss(args, negative_ratio=2., negatives_for_hard=100, alpha=1.)
    model = utils.model.load_from_checkpoint(os.path.join(args.model_dir, 'checkpoints', args.fmodel), losses)
    model.summary()

    # plot loss
    history = pickle.load(open(os.path.join(args.model_dir, 'train_history.pkl'), 'rb'))
    plot_performance(history, args.saveto)

    # Quiver package server
    # print(global_variables.IMAGES_PATH)
    # visualize_conv2D_feat_maps(model, args.model_dir, global_variables.IMAGES_PATH)
    # exit()

    # crop images into (?, ?, 3)
    l_images, l_bboxes, l_tbboxes, l_classes, l_tclasses, l_scores = [], [], [], [], [], []
    keys = list(gt.keys())
    for i, k in tqdm(enumerate(sorted(keys)), desc='[INFO] Detecting objects on images...', total=len(keys)):

        # get ground truth
        image = (np.array(misc.imread(os.path.join(global_variables.IMAGES_PATH, k + '.jpg'))) / 255.) * 2 - 1
        boxes, objclass = gt[k]

        # crop image
        patches_of_image, _, xy = utils.data.preprocessing.crop_to_images(image, boxes, objclass,
                                                                          size=args.ssd_params.img_shape[0],
                                                                          return_coord=True)
        # predict on image crops
        classes, scores, bboxes = np_detect_objects(model, ssd_net, patches_of_image, args)

        # prediction on full image
        classes, scores, bboxes = to_original_coordinates(image, classes, scores, bboxes, xy, args)

        # append results for image i
        l_images.append(np.array(image))
        l_bboxes.append(np.array(bboxes))
        l_classes.append(np.array(classes))
        l_scores.append(np.array(scores))
        l_tbboxes.append(to_relative_coordinates(boxes, image.shape)
                         if np.array(boxes).any()
                         else np.zeros((0, 4), dtype='float32'))
        l_tclasses.append(np.int64(np.argmax(objclass, axis=1) + 1)
                          if np.array(boxes).any()
                          else np.zeros((0,), dtype='int64'))


    # --------------------------------------------------------------------- ------------------------------------------#
    #  Evaluation: compute TP, FP and FN
    #
    #  Return a dictionary structured as:
    #     {
    #         'image-0001 (filename)': {
    #
    #             '0.01 (threshold)': {
    #
    #                 '1 (class)': {
    #                     'TP': 0,
    #                     'FP': 0,
    #                     'FN': 0
    #                 },
    #
    #                 '2 (class)': {
    #                     'TP': 0,
    #                     'FP': 0,
    #                     'FN': 0
    #                 },
    #
    #                 'cm (confusion matrix)': {
    #                     [0, 0, 0,
    #                      0, 0, 0,
    #                      0, 0, 0]
    #                 }
    #             },
    #             '0.02 (threshold)': ...,
    #
    #             ...
    #
    #         }
    #
    #         'image-0001 (filename)': ...,
    #
    #         ...
    #     }
    #
    # --------------------------------------------------------------------- ------------------------------------------#

    # save confidence values to guide evaluation (sample vector)
    unique_values = np.unique(np.hstack(l_scores))
    confidences = np.sort(unique_values)[::3]

    # match gt/p according to predictions
    y_true, y_scores, y_pred = match_truth_and_predictions(batch_of_bb=l_bboxes,
                                                           batch_of_cls=l_classes,
                                                           batch_of_bbgt=l_tbboxes,
                                                           batch_of_clsgt=l_tclasses,
                                                           batch_of_sc=l_scores,
                                                           include_negatives=True,
                                                           args=args)

    # compute error types across test set
    dict_of_err = {k: {} for k in sorted(keys)}
    for t in tqdm(confidences, desc='[INFO] .. at cut-off t..'):

        # compute error types I, II and III for the entire batch of images
        for k, true, scores, predictions in zip(keys, y_true, y_scores, y_pred):

            # take only predictions above t
            predictions_at_t = np.where(scores > t, predictions, np.zeros_like(predictions))

            # compute counts (confusion matrix)
            cm = np.nan_to_num(confusion_matrix(true,
                                                predictions_at_t,
                                                labels=list(range(global_variables.NUM_CLASSES))))

            # count TP, FP, FN per class
            dict_of_err[k][t] = {1: {}, 2: {}, 'cm': 0}
            dict_of_err[k][t]['cm'] = cm
            for c in [1, 2]:
                TP = cm[c][c]
                FN = cm[c][0] + cm[c][2 if c == 1 else 1]
                FP = cm[0][c] + cm[2 if c == 1 else 1][c]

                # store it
                dict_of_err[k][t][c]['tp'] = TP
                dict_of_err[k][t][c]['fn'] = FN
                dict_of_err[k][t][c]['fp'] = FP

    # dump it in a file
    f = open(os.path.join(args.saveto, 'I-III_ErrTypes_NMS_{}.pickle'.format(
        args.eval_params.nms_threshold)), 'wb')
    pickle.dump((dict_of_err, confidences), f)
    f.close()

    # ----------------------------------------------------------------------------------------------------------------#
    #
    # Evaluation: precision-recall curves, froc analysis, confusion matrix
    #
    # ----------------------------------------------------------------------------------------------------------------#

    styles = ['solid', 'dashed']
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for c in [1, 2]:
        cumsumTPs, cumsumFPs, cumsumFNs = [], [], []
        for t in tqdm(confidences[::-1]):
            TP, FP, FN, gt = 0, 0, 0, 0
            # cm = np.zeros((3, 3))

            for k in dict_of_err:
                TP += dict_of_err[k][t][c]['tp']
                FP += dict_of_err[k][t][c]['fp']
                FN += dict_of_err[k][t][c]['fn']
                gt += dict_of_err[k][t][c]['tp'] + dict_of_err[k][t][c]['fn']
                # cm += dict_of_err[k][t]['cm']
            # TP = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
            # FP = cm[0][1] + cm[0][2]
            # FN = cm[1][0] + cm[2][0]
            # gt = TP + FN
            cumsumTPs.append(TP)
            cumsumFPs.append(FP)
            cumsumFNs.append(FN)

        fps = np.asarray(cumsumFPs) / float(len(list(dict_of_err.keys())))
        # fns = np.asarray(cumsumFNs) / np.float32(np.asarray(cumsumTPs) + np.asarray(cumsumFNs))
        recall = np.asarray(cumsumTPs) / float(gt)
        precision = np.nan_to_num(np.asarray(cumsumTPs) / np.float32(np.asarray(cumsumTPs) + np.asarray(cumsumFPs)))
        precision = np.clip(precision, a_min=0, a_max=1)
        ap, precision, recall = compute_average_precision(precision, recall)

        # FROC curve per class
        fps = np.concatenate([[0], fps, [10]])
        f = UnivariateSpline(recall, fps, k=3)
        ynew = np.linspace(0.0, 1.0, 100, endpoint=True)
        ax1.plot(f(ynew), ynew, linestyle=styles[c - 1])
        ax1.set_xlabel('False Positives per Image')
        ax1.set_ylabel('Detection rate')
        ax1.legend(loc='lower right')

        # precision-recall curve per class
        f = UnivariateSpline(recall, precision, k=3)
        xnew = np.linspace(min(recall), max(recall), 100)
        ax2.step(xnew, f(xnew), linestyle=styles[c - 1], label='AP={0:0.4f}'.format(ap))
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc='lower left')

        # print relevant evaluation info
        recall = recall[:-1]
        precision = precision[:-1]
        # cut_off = np.argmin(recall - precision)
        cut_off = confidences[np.argmin(np.abs(confidences - 0.5))]
        print('[SUMMARY]')
        print('Class', c)
        print('Threshold cut-off', cut_off)
        print('Recall@AveragePrecision',
              np.where(np.asarray(precision) - ap < 0.001, recall, np.zeros_like(recall)).max())
        print('Average Precision', ap)

        # confusion matrix
        for k in dict_of_err:
            cm += dict_of_err[k][cut_off]['cm']
        cm /= len(dict_of_err)
        fig3 = plot_confusion_matrix(cm, ['BG', 'PF', 'WBC'], cmap='PuBu', normalize=True)

    fig1.savefig(os.path.join('./', 'froc-curve.png'))
    fig1.show()
    fig2.savefig(os.path.join('./', 'precision-recall-curve.png'))
    fig2.show()
    fig3.savefig(os.path.join('./', 'confusion-matrix-class.png'))
    fig3.show()

    # draw predicted and target bboxes on images in test set
    draw_predicted_and_target_bboxes_on_images(l_images,
                                               {'true': (l_tclasses, l_tclasses, l_tbboxes),
                                                'pred': (l_classes, l_scores, l_bboxes)},
                                               args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate SSD Multibox to detect Malaria.')

    parser.add_argument('--gt', dest='gt', help='path to ground truth pickled file')
    parser.add_argument('--model', dest='fmodel', help='model file (json)')
    parser.add_argument('--model_dir', dest='model_dir', help='path to model directory')
    parser.add_argument('--output_dir', dest='saveto', help='path to save results')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=8, help='batch size')
    parser.add_argument('--matching_thrs', type=float, dest='matching_threshold', default=0.5, help='jaccard overlap')
    parser.add_argument('--eval_params', dest='eval_params', help='Evaluation parameters')
    parser.add_argument('--ssd_params', dest='ssd_params', help='SSD multibox parameters')

    args = parser.parse_args()

    if not args.gt:
        parser.error('gt file required')

    if not args.fweights or args.fmodel:
        parser.error('need to define a model to evaluate (either weights or json file were missing).')

    global_variables.init()
    evaluate_function(args)
