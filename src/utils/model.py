import os

import numpy as np
import tensorflow as tf
from quiver_engine import server

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.models import model_from_json

from tensorflow.python.client import device_lib

from utils.tensors import get_shape


def get_available_gpus(ngpus=-1):
    '''
    :param int ngpus: GPUs max to use. Default -1 means all gpus.
    :returns: List of gpu devices. Ex.: ['/gpu:0', '/gpu:1', ...]
    '''
    local_device_protos = device_lib.list_local_devices()
    gpus_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpus_list[:ngpus] if ngpus > -1 else gpus_list

def load_from_checkpoint(fmodel, multiboxloss):
    """
        Load model from checkpoint.

    :param fmodel: (string) path to model config file.
    :param multiboxloss: (obj) loss class object

    :return: (obj) keras model object
    """

    # load json and create model
    with open(fmodel, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json, custom_objects={'Normalize': Normalize,
                                                        'tf': tf, # keras workaround when working with Lambda layers
                                                        'confidence_loss': multiboxloss.confidence_loss,
                                                        'localisation_loss': multiboxloss.localisation_loss})
    # load weights into new model
    base, filename = os.path.split(fmodel)

    model.load_weights(os.path.join(base, 'weights.' + filename.split('.', 1)[-1][:-4] + 'hdf5'), by_name=True)
    print('[INFO] Model loaded from disk')

    return model


def visualize_conv2D_feat_maps(model, model_dir, data_dir):

    quiver_dir = os.path.join(model_dir, 'quiver')
    if not os.path.exists(quiver_dir):
        os.makedirs(quiver_dir)

    server.launch(
        model,  # a Keras Model

        # classes=['background', 'falciparum'],

        # where to store temporary files generatedby quiver (e.g. image files of layers)
        temp_folder=quiver_dir,

        # a folder where input images are stored
        input_folder=data_dir,

        # the localhost port the dashboard is to be served on
        port=5005
    )


def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)
    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])

        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)

        return jaccard

    def condition(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """

        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """

        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores, feat_ymin, \
     feat_xmin, feat_ymax, feat_xmax] = tf.while_loop(condition,
                                                      body,
                                                      [i, feat_labels, feat_scores,
                                                       feat_ymin, feat_xmin,
                                                       feat_ymax, feat_xmax])

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]

    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)

    return feat_labels, feat_localizations, feat_scores


def np_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               ignore_threshold=0.4,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=np.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = np.maximum(ymin, bbox[0])
        int_xmin = np.maximum(xmin, bbox[1])
        int_ymax = np.minimum(ymax, bbox[2])
        int_xmax = np.minimum(xmax, bbox[3])

        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if not union_vol.all():
            idx = np.where(union_vol == 0.0)[0]
            union_vol[idx] += 1e-12

        return inter_vol / union_vol

    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - (href / 2.)
    xmin = xref - (wref / 2.)
    ymax = yref + (href / 2.)
    xmax = xref + (wref / 2.)
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = np.zeros(shape, dtype=np.int32)
    feat_scores = np.zeros(shape, dtype=dtype)
    feat_ymin = np.zeros(shape, dtype=dtype)
    feat_xmin = np.zeros(shape, dtype=dtype)
    feat_ymax = np.ones(shape, dtype=dtype)
    feat_xmax = np.ones(shape, dtype=dtype)

    for label, bbox in zip(labels, bboxes):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        jaccard = jaccard_with_anchors(bbox)

        # Mask: check threshold + scores + no annotations + num_classes.
        mask = jaccard >= feat_scores
        mask = np.logical_and(mask, jaccard >= ignore_threshold)
        # mask = np.logical_and(mask, feat_scores > -0.5)
        # mask = np.logical_and(mask, label < num_classes)

        # Update values using mask.
        feat_labels = mask.astype(np.int32) * label + (1 - mask.astype(np.int32)) * feat_labels
        feat_scores = np.where(mask.astype(dtype), jaccard, feat_scores)

        # Update values using mask.
        feat_ymin = mask.astype(dtype) * bbox[0] + (1 - mask.astype(dtype)) * feat_ymin
        feat_xmin = mask.astype(dtype) * bbox[1] + (1 - mask.astype(dtype)) * feat_xmin
        feat_ymax = mask.astype(dtype) * bbox[2] + (1 - mask.astype(dtype)) * feat_ymax
        feat_xmax = mask.astype(dtype) * bbox[3] + (1 - mask.astype(dtype)) * feat_xmax

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = np.log(feat_h / href) / prior_scaling[2]
    feat_w = np.log(feat_w / wref) / prior_scaling[3]

    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = np.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)

    return feat_labels, feat_localizations, feat_scores


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores


def np_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=np.float32):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    target_labels = []
    target_localizations = []
    target_scores = []
    for i, anchors_layer in enumerate(anchors):
        t_labels, t_loc, t_scores = np_ssd_bboxes_encode_layer(labels, bboxes,
                                                               anchors_layer,
                                                               num_classes,
                                                               ignore_threshold,
                                                               prior_scaling, dtype)

        target_labels.append(np.reshape(t_labels, newshape=(-1)))
        target_localizations.append(np.reshape(t_loc, newshape=(-1, 4)))
        target_scores.append(np.reshape(t_scores, newshape=(-1)))

    return np.hstack(target_labels)[:, np.newaxis], \
           np.vstack(target_localizations), \
           np.hstack(target_scores)[:, np.newaxis]


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])

    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               detection_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      detection_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    detection_threshold = 0.0 if detection_threshold is None else detection_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, detection_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         detection_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      detection_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        detection_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
                                           detection_threshold=None):
    """Extract classes, scores and bounding boxes from features in one layer.
     Batch-compatible: inputs are supposed to have batch-type shapes.

     Args:
       predictions_layer: A SSD prediction layer;
       localizations_layer: A SSD localization layer;
      detection_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
     Return:
      classes, scores, bboxes: Input Tensors.
     """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = get_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = get_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))

    # Boxes selection: use threshold or score > no-label criteria.
    if detection_threshold is None or detection_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        classes = tf.argmax(sub_predictions, axis=2) + 1
        scores = tf.reduce_max(sub_predictions, axis=2)
        # Only keep predictions higher than threshold.
        mask = tf.greater(scores, detection_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)

    # Assume localization layer already decoded.
    bboxes = localizations_layer
    return classes, scores, bboxes


def tf_ssd_bboxes_select_all_classes(predictions_net, localizations_net,
                                     detection_threshold=None,
                                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      detection_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],
                                                       localizations_net[i],
                                                       detection_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)

        return classes, scores, bboxes


def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


def to_relative_coordinates(boxes, shape):
    '''
        Absolute pixel values into relative ones.
        Also, reordering of coordinates to match tf implementation.

    :param boxes: numpy array. coordinates of bounding boxes (N x 4)
    :param shape: tuple. image shape.

    :return: (ymin, xmin, ymax, xmax)
    '''

    ymin, xmin, ymax, xmax = boxes.T

    xmin = xmin / float(shape[1])
    xmax = xmax / float(shape[1])
    ymin = ymin / float(shape[0])
    ymax = ymax / float(shape[0])

    return np.concatenate([ymin[:, None], xmin[:, None], ymax[:, None], xmax[:, None]], axis=-1)


def np_ssd_bboxes_decode(feat_localizations,
                         anchor_bboxes,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.
    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    batch_of_bboxes = []
    for feat_localization in feat_localizations:
        # Reshape for easier broadcasting.
        l_shape = feat_localization.shape
        feat_localization = np.reshape(feat_localization,
                                        (-1, l_shape[-2], l_shape[-1]))
        yref, xref, href, wref = anchor_bboxes
        xref = np.reshape(xref, [-1, 1])
        yref = np.reshape(yref, [-1, 1])

        # Compute center, height and width
        cx = feat_localization[:, :, 0] * wref * prior_scaling[0] + xref
        cy = feat_localization[:, :, 1] * href * prior_scaling[1] + yref
        w = wref * np.exp(feat_localization[:, :, 2] * prior_scaling[2])
        h = href * np.exp(feat_localization[:, :, 3] * prior_scaling[3])
        # bboxes: ymin, xmin, xmax, ymax.
        bboxes = np.zeros_like(feat_localization)
        bboxes[:, :, 0] = cy - h / 2.
        bboxes[:, :, 1] = cx - w / 2.
        bboxes[:, :, 2] = cy + h / 2.
        bboxes[:, :, 3] = cx + w / 2.

        # Back to original shape.
        bboxes = np.reshape(bboxes, l_shape)
        batch_of_bboxes.append(bboxes)

    return np.array(batch_of_bboxes)
    # return bboxes


def np_ssd_bboxes_select_layer(predictions_layer,
                               localizations_layer,
                               anchors_layer,
                               detection_threshold=0.5,
                               decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.
    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        localizations_layer = np_ssd_bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if detection_threshold is None or detection_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes, scores, bboxes = [], [], []
        for predictions_i, localizations_i in zip(predictions_layer, localizations_layer):
            classes_i = np.argmax(predictions_i, axis=-1)
            scores_i = np.amax(predictions_i, axis=-1)
            mask = classes_i > 0
            classes.append(classes_i[mask])
            scores.append(scores_i[mask])
            bboxes.append(localizations_i[mask])
    else:
        classes, scores, bboxes = [], [], []
        for predictions_i, localizations_i in zip(predictions_layer[:, :, 1:], localizations_layer):
            idxes = np.where(predictions_i > detection_threshold)
            classes.append(idxes[-1] + 1)
            scores.append(predictions_i[idxes])
            bboxes.append(localizations_i[idxes[0]])

    return classes, scores, bboxes


def np_ssd_bboxes_select(predictions_net,
                         localizations_net,
                         anchors_net,
                         detection_threshold=0.5,
                         decode=True):
    """Extract classes, scores and bounding boxes from network output layers.
    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = np_ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            detection_threshold, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

    # convert into (BATCH, N, 1|4) format
    rclasses, rscores, rbboxes = [], [], []
    for i in range(len(predictions_net[0])):
        classes_i, scores_i, bboxes_i = [], [], []
        for layer in range(len(predictions_net)):
            classes_i.append(l_classes[layer][i])
            scores_i.append(l_scores[layer][i])
            bboxes_i.append(l_bboxes[layer][i])

        rclasses.append(classes_i)
        rscores.append(scores_i)
        rbboxes.append(bboxes_i)

    return rclasses, rscores, rbboxes

