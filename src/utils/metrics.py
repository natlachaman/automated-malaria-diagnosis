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
"""TF Extended: additional metrics.
"""
import tensorflow as tf
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


# ============================================================================ #
# TF Extended metrics: Loss function
# ============================================================================ #
class MultiboxLoss(object):

    def __init__(self, args, negative_ratio=2., negatives_for_hard=100, alpha=1.):
        self.matching_threshold  = args.matching_threshold
        self.negative_ratio      = negative_ratio
        self.negatives_for_hard  = negatives_for_hard
        self.alpha               = alpha
        self.batch_size          = args.batch_size
        self.num_classes         = args.ssd_params.num_classes

    @staticmethod
    def _abs_smooth(x):
        """Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        We use here a differentiable definition using min(x) and abs(x). Clearly
        not optimal, but good enough for our purpose!
        """
        # absx = tf.abs(x)
        # minx = tf.minimum(absx, 1)
        # r = 0.5 * ((absx - 1) * minx + absx)

        absx = tf.abs(x)
        r = array_ops.where(absx < 1, math_ops.square(x) / 2.0, absx - 0.5)

        return r

    def confidence_loss(self, y_true, y_pred):
        gclasses = tf.cast(y_true[:, :, 0], 'int32')
        gscores = y_true[:, :, -1]
        logits = y_pred

        dtype = logits.dtype
        batch_loss = []
        for i in range(self.batch_size):

            # Compute positive matching mask...
            pmask = gscores[i] > self.matching_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)

            # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = tf.nn.softmax(logits[i])
            nmask = tf.logical_and(tf.logical_not(pmask),
                                   gscores[i] > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask,
                               predictions[:, 0],
                               1. - fnmask)
            # nvalues_flat = tf.reshape(nvalues, [-1])

            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(self.negative_ratio * n_positives, tf.int32)  # + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)
            # in case there are no positives...
            has_negatives = tf.cast(tf.greater(n_neg, 0), tf.int32)
            n_neg = n_neg + (1 - has_negatives) * self.negatives_for_hard

            # final negative mask
            val, idxes = tf.nn.top_k(-nvalues, k=n_neg)
            max_hard_pred = -val[-1]
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            # neg_idx = tf.random_shuffle(tf.range(0, tf.shape(nvalues_flat)[0]))
            # neg_idx = tf.boolean_mask(neg_idx, nmask)[:n_neg]
            # smask = tf.zeros_like(nvalues_flat, dtype=tf.int32)
            # smask[neg_idx] = 1
            # nmask = tf.logical_and(nmask, smask)
            # fnmask = tf.cast(nmask, dtype)

            with tf.name_scope('cross_entropy_pos') as scope:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                      labels=gclasses[i])
                cross_pos = tf.losses.compute_weighted_loss(loss, fpmask, scope=scope)

            with tf.name_scope('cross_entropy_neg') as scope:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                      labels=no_classes)
                cross_neg = tf.losses.compute_weighted_loss(loss, fnmask, scope=scope)

            batch_loss.append(tf.add_n([cross_neg, cross_pos]))

        return tf.div(tf.add_n(batch_loss), self.batch_size, name='total_cross_entropy')

    def localisation_loss(self, y_true, y_pred):
        glocalisations = y_true[:, :, :-1]
        gscores = y_true[:, :, -1]
        localisations = y_pred[:, :, :-1]

        dtype = localisations.dtype
        batch_loss = []
        for i in range(self.batch_size):
            # Compute positive matching mask...
            pmask = gscores[i] > self.matching_threshold
            fpmask = tf.cast(pmask, dtype)

            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization') as scope:
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(self.alpha * fpmask, axis=-1)
                loss = self._abs_smooth(localisations[i] - glocalisations[i])
                l1_loc = tf.losses.compute_weighted_loss(loss, weights, scope=scope)

            batch_loss.append(l1_loc)

        # total loss for batch
        return tf.div(tf.add_n(batch_loss), self.batch_size, name='total_loc_loss')


# ============================================================================ #
# TF Extended metrics: Tensorflow Object Detection API metrics
# ============================================================================ #
def compute_precision_recall(scores, predictions, ground_truth, cls, num_samples):
    """Compute precision and recall.
    Args:
    scores: A float numpy array representing detection score
    labels: A boolean numpy array representing true/false positive labels
    num_gt: Number of ground truth instances
    num_samples: Number of testing images
    Raises:
    ValueError: if the input is not of the correct format
    Returns:
    precision: Fraction of positive instances over detected ones. This value is
      None if no ground truth labels are present.
    recall: Fraction of detected positive instance over all positive instances.
      This value is None if no ground truth labels are present.
    """

    def true_positives_false_positives_labels(y_scores, y_pred, y_true, cls):
        class_idx = np.where(y_pred == cls)[0]
        scores, true = y_scores[class_idx], y_true[class_idx]
        labels = np.zeros_like(scores)
        labels[true == cls] = 1
        return scores, labels

    scores, labels = true_positives_false_positives_labels(scores, predictions, ground_truth, cls)

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)

    precision = cum_true_positives.astype(float) / (cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / np.sum(np.int32(ground_truth == cls))

    mean_false_positives = cum_false_positives / num_samples

    return precision, recall, mean_false_positives


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

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return average_precision


def jaccard_distance(bbox_ref, bboxes):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with Jaccard scores.
    """
    # Should be more efficient to first transpose.
    ymin, xmin, ymax, xmax = bboxes.T

    # Intersection bbox and volume.
    int_ymin = np.maximum(ymin, bbox_ref[0])
    int_xmin = np.maximum(xmin, bbox_ref[1])
    int_ymax = np.minimum(ymax, bbox_ref[2])
    int_xmax = np.minimum(xmax, bbox_ref[3])

    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)

    # Volumes.
    inter_vol = h * w
    area = (ymax - ymin) * (xmax - xmin)
    area_ref = (bbox_ref[3] - bbox_ref[1]) * (bbox_ref[2] - bbox_ref[0])
    union_vol = -inter_vol + area + area_ref
    jaccard = np.nan_to_num(inter_vol / union_vol)

    return jaccard


def match_truth_and_predictions_single_batch(args, include_negatives=False, **kwargs):
    '''
        Get one prediction per ground truth annotation per single batch (image).
    :param bbgt: (numy array)  ground truth boxes (num_of_boxes, 4)
    :param clsgt: (numpy array) ground truth object class (num_of_boxes, 1)
    :param bb: (numpy array)  predicted boxes (num_of_boxes, 4)
    :param cls: (numpy array) predicted object class (num_of_boxes, 1)
    :param matching_thrs: (float) minimum overlap (jaccard) between pred and true boxes considered a "hit"

    :return: y_pred: (list) with class predictions (int) of the same length as 'keep_to_k'.
             y_true: (list) with ground truth class (int) of the same length as 'keep_to_k'.
    '''

    tbboxes = kwargs.get('tbboxes', None)
    tclasses = kwargs.get('tclasses', None)
    bboxes = kwargs.get('bboxes', None)
    classes = kwargs.get('classes', None)
    scores = kwargs.get('scores', None)
    threshold = args.matching_threshold

    y_pred, y_true, y_scores = [], [], []
    for tbboxes_i, tclasses_i in zip(tbboxes, tclasses):
        # add positive annotation
        y_true.append(tclasses_i)

        # are there any predictions at all?
        if np.any(bboxes):

            # was this annotation detected?
            jaccard = jaccard_distance(tbboxes_i, bboxes)
            hit = np.any(jaccard > threshold)

            # if there are no hits, continue to the next annotation
            if not hit:
                y_pred.append(0)
                y_scores.append(np.float32(0.0))
                continue

            # was it correctly detected?
            class_mask = np.logical_and(jaccard > threshold, classes == tclasses_i)
            class_hit = np.sum(np.int32(class_mask))

            # compute the best match within the correct class. If it wasn't correctly detected,
            # return the best match of an incorrect class
            condition = class_mask if class_hit > 0 else jaccard > threshold
            best_match = np.where(condition, jaccard, 0).max(keepdims=1) == jaccard
            y_pred.append(classes[best_match][0])
            y_scores.append(scores[best_match][0])

            # delete the best match from list (to avoid duplicated counts)
            bboxes = np.delete(bboxes, np.where(best_match)[0], axis=0)
            classes = np.delete(classes, np.where(best_match)[0])
            scores = np.delete(scores, np.where(best_match)[0])

        else:
            # if there are no predictions
            y_pred.append(0)
            y_scores.append(np.float32(0.0))

    # if there are non-matched detections left, count them as FN
    if np.any(bboxes):
        y_true.extend([0]*len(bboxes))
        y_pred.extend([int(i) for i in classes])
        y_scores.extend([float(i) for i in scores])

    # fill up both lists with zeros (true negatives) up to 'keep_top_k'
    if include_negatives:
        y_true.extend([0]*(args.eval_params.keep_top_k - len(y_true)))
        y_pred.extend([0]*(args.eval_params.keep_top_k - len(y_pred)))
        y_scores.extend([1.0]*(args.eval_params.keep_top_k - len(y_scores)))

    return y_true, y_scores, y_pred


def match_truth_and_predictions(args, **kwargs):
    """
            Get one prediction per ground truth annotation. Batch mode.

        :param batch_of_bbgt: (numpy array) ground truth boxes (batch_size, num_of_boxes, 4)
        :param batch_of_clsgt: (numpy array) ground truth object class categories (batch_size, num_of_boxes, 1)
        :param batch_of_bb: (numpy array) predicted boxes (batch_size, num_of_boxes, 4)
        :param batch_of_cls: (numpy array) predicted object class categories (batch_size, num_of_boxes, 1)
        :param matching_thrs: (float) acceptance criteria - minimum jaccard overlap, values between (0, 1)

        :return: y_pred: (list) with class predictions (int) of the same length as gt annotations.
                 y_true: (list) with ground truth class (int) of the same length as gt annotations.
    """
    b_tbboxes = kwargs.get('batch_of_bbgt', None)
    b_tclasses = kwargs.get('batch_of_clsgt', None)
    b_bboxes = kwargs.get('batch_of_bb', None)
    b_classes = kwargs.get('batch_of_cls', None)
    b_scores = kwargs.get('batch_of_sc', None)
    include_negatives = kwargs.get('include_negatives', False)
    confidence = kwargs.get('conf_threshold', 0.0)

    if b_tbboxes is None or b_tclasses is None:
        TypeError('missing ground truth argument')

    if b_bboxes is None or b_classes is None or b_scores is None:
        TypeError('missing predicted argument')

    y_true, y_pred, y_scores = [], [], []
    i = 0
    for tbboxes, tclasses, bboxes, classes, scores in zip(b_tbboxes, b_tclasses, b_bboxes, b_classes, b_scores):

        # confidence mask
        mask = np.where(scores > confidence)[0]
        if len(bboxes) > 0:
            bboxes = bboxes[mask, :]
            classes = classes[mask]
            scores = scores[mask]

        t, s, p = match_truth_and_predictions_single_batch(args, include_negatives,
                                                           tbboxes=tbboxes,
                                                           tclasses=tclasses,
                                                           bboxes=bboxes,
                                                           classes=classes,
                                                           scores=scores)
        y_true.append(t)
        y_pred.append(p)
        y_scores.append(s)

    return y_true, y_scores, y_pred


def nms_for_single_image(boxes, threshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.array([]), np.array([])

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1, x1, y2, x2 = np.vstack(boxes).T
    boxes = np.vstack(boxes)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[-1]
        pick.append(i)

        overlap = jaccard_distance(boxes[i, :], boxes[idxs[:last], :])

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked
    return np.vstack(boxes)[pick], pick