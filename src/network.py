# Copyright 2016 Paul Balanca. All Rights Reserved.
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
#  ==============================================================================
import sys
sys.path.append('../')
import os


import keras
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization, ZeroPadding2D, Conv2DTranspose
from keras.layers import Concatenate, Input, Lambda, Activation
from keras.layers.merge import Multiply


import math
import numpy as np
import tensorflow as tf
from collections import namedtuple

import utils
import global_variables
global_variables.init()


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
def ssd_params_vars():
    return namedtuple('SSDParameters', ['img_shape',
                                        'num_classes',
                                        'no_annotation_label',
                                        'feat_layers',
                                        'feat_shapes',
                                        'anchor_size_bounds',
                                        'anchor_sizes',
                                        'anchor_ratios',
                                        'anchor_steps',
                                        'anchor_offset',
                                        'normalizations',
                                        'prior_scaling',
                                        'matching_threshold'])


def eval_params_vars():
    return namedtuple('EvaluationParameters', ['detection_threshold',
                                               'nms_threshold',
                                               'select_top_k',
                                               'keep_top_k'])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    SSDParams = ssd_params_vars()
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['conv2_1', 'conv3_1', 'conv3_3', 'conv4_2', 'conv4_3', 'conv5_3'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[(2, .5),
                       (2, .5, 3, 1./3),
                       (2, .5, 3, 1./3),
                       (2, .5, 3, 1./3),
                       (2, .5),
                       (2, .5)],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2],
        matching_threshold=0.5)

    EvalParams = eval_params_vars()
    default_eval = EvalParams(detection_threshold=0.5,
                              nms_threshold=0.5,
                              select_top_k=400.,
                              keep_top_k=200.)

    def __init__(self, args=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if hasattr(args, 'ssd_params') or args.ssd_params is not None:
            self.params = args.ssd_params
        else:
            self.params = SSDNet.default_params

        if hasattr(args, 'eval_params'):
            self.eval = args.eval_params
        else:
            self.eval = SSDNet.default_eval

        self.args = args

    def net(self):

        """SSD network definition.
        """
        return ssd_net(self.params.img_shape,
                       num_classes=self.params.num_classes,
                       feat_layers=self.params.feat_layers,
                       anchor_sizes=self.params.anchor_sizes,
                       anchor_ratios=self.params.anchor_ratios,
                       normalizations=self.params.normalizations,
                       args=self.args)

    def anchors(self, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(self.params.img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes (tf)
        """
        return utils.model.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=self.params.matching_threshold,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_encode_numpy(self, labels, bboxes, anchors):
        """Encode labels and bounding boxes (numpy)
        """
        return utils.model.np_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            ignore_threshold=self.params.matching_threshold,
            prior_scaling=self.params.prior_scaling)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Decode labels and bounding boxes (tf).
        """
        return utils.model.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode_numpy(self, feat_localizations, anchors):
        """Decode labels and bounding boxes (numpy).
        """
        return utils.model.np_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling)

    def bboxes_detect(self, y_pred):
        """Get the detected bounding boxes from the SSD network output.

        """
        # Reshape network's output in the original format
        p = y_pred[0]
        l = y_pred[1][:, :, :-1]  # related to keras-2.1.0 target shape match issue (see SSD def. below)

        predictions, localisations = [], []
        start = 0
        for i, (m, n) in enumerate(self.params.feat_shapes):
            num_anchors = len(self.params.anchor_ratios[i]) + len(self.params.anchor_sizes[i])

            predictions.append(tf.reshape(p[:, start:start + (m * n * num_anchors), :],
                                          shape=(-1, m, n, num_anchors, self.params.num_classes)))
            localisations.append(tf.reshape(l[:, start:start + (m * n * num_anchors), :],
                                            shape=(-1, m, n, num_anchors, 4)))
            start += (m * n * num_anchors)

        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):

            # Decode boxes
            localisations = self.bboxes_decode(localisations, self.anchors())

            # Select top_k bboxes from predictions
            rscores, rbboxes = utils.model.tf_ssd_bboxes_select(predictions, localisations,
                                                                detection_threshold=self.eval.detection_threshold,
                                                                num_classes=self.params.num_classes)

            # Sort them in descending order.
            rscores, rbboxes = utils.bboxes.bboxes_sort(rscores, rbboxes, top_k=self.eval.select_top_k)

            # Apply NMS algorithm.
            rscores, rbboxes = utils.bboxes.bboxes_nms_batch(rscores, rbboxes,
                                                             nms_threshold=self.eval.nms_threshold,
                                                             keep_top_k=self.eval.keep_top_k)

            return rscores, rbboxes

    def bboxes_detect_numpy(self, y_pred, apply_nms=True):
        """Get the detected bounding boxes from the SSD network output.

        """
        # Reshape network's output in the original format: (layers, batch, anchors, classes|4)
        p = y_pred[0]
        l = y_pred[1][:, :, :-1] # related to keras-2.1.0 target shape match issue (see SSD def. below)

        predictions, localisations = [], []
        start = 0
        for i, (m, n) in enumerate(self.params.feat_shapes):
            num_anchors = len(self.params.anchor_ratios[i]) + len(self.params.anchor_sizes[i])

            predictions.append(np.reshape(p[:, start:start + (m * n * num_anchors), :],
                                          newshape=(-1, m, n, num_anchors, self.params.num_classes)))
            localisations.append(np.reshape(l[:, start:start + (m * n * num_anchors), :],
                                            newshape=(-1, m, n, num_anchors, 4)))
            start += (m * n * num_anchors)

        # Decode and select top_k bboxes from predictions
        classes, scores, bboxes = utils.model.np_ssd_bboxes_select(predictions,
                                                                   localisations,
                                                                   self.anchors(),
                                                                   detection_threshold=self.eval.detection_threshold)
        # Sort them in descending order.
        classes, scores, bboxes = utils.bboxes.np_bboxes_sort(classes,
                                                              scores,
                                                              bboxes,
                                                              top_k=self.eval.select_top_k)

        # Apply NMS algorithm. Optional: when evaluating full images
        if apply_nms:
            classes, scores, bboxes = utils.bboxes.np_bboxes_nms(classes,
                                                                 scores,
                                                                 bboxes,
                                                                 nms_threshold=self.eval.nms_threshold)

        return classes, scores, bboxes

# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.


    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]
    img_size = img_shape[0]

    # Compute object scales
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    scales, step = np.linspace(min_ratio, max_ratio, n_feat_layers, retstep=True, dtype='int32')

    # Assign the (min, max) to each feat_layers
    sizes = []
    for i, ratio in enumerate(scales):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + round(step, 2)) / 100.))

    return sizes


def ssd_feat_shapes_from_net(model, keyword, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shape = list()
    for layer in model.layers:
        if keyword in layer.name:
            feat_shape.append(layer.output_shape[1:-1])
    return feat_shape


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.

    """
    # Compute the position grid: simple way.
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) / feat_shape[0]
    x = (x.astype(dtype) + offset) / feat_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(ratios) + len(sizes)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)

    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD and DSSD
# =========================================================================== #
def ssd_layers_specs():
    """
        Layer specifications of SSD meta-network.
    """
    return {'block6': {'filters': 1024,
                       'kernel_size': (3, 3),
                       'padding': 'same'},

            'block7': {'filters': 1024,
                       'kernel_size': (1, 1),
                       'padding': 'same'},

            'block8': {'filters': (256, 512),
                       'kernel_size': [(1, 1), (3, 3)],
                       'kernel_stride': (2, 2)},

            'block9': {'filters': (128, 256),
                       'kernel_size': [(1, 1), (3, 3)],
                       'kernel_stride': (2, 2)},

            'block10': {'filters': (128, 256),
                        'kernel_size': [(1, 1), (3, 3)],
                        'kernel_stride': (2, 2)},

            'block11': {'filters': (128, 256),
                        'kernel_size': [(1, 1), (3, 3)],
                        'kernel_stride': (2, 2)},

            'block12': {'filters': (128, 256),
                        'kernel_size': [(1, 1), (4, 4)],
                        'kernel_stride': (2, 2)}

            }


def ssd_multibox_layer(net,
                       layer,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1):
    """Construct a multibox layer, return a class and localization predictions.
    """
    base_layer = net[layer]

    # DSSD: added sequence
    # x = Conv2D(256, [1, 1], padding='same')(base_layer)
    # x = Conv2D(256, [1, 1], padding='same')(x)
    # x = Conv2D(1024, [1, 1], padding='same')(x)
    # y = Conv2D(1024, [1, 1], padding='same')(base_layer)
    # base_layer = Add()([x, y])

    # Spatial L2 norm (for exploiting gradients)
    if normalization > 0:
        base_layer = Normalize(20, name='norm_{}'.format(layer))(base_layer)

    # Number of anchors
    num_anchors = len(ratios) + len(sizes)

    # Location prediction.
    loc_pred = Conv2D(num_anchors * 4, [3, 3], padding='same', name='loc_{}'.format(layer))(base_layer)
    shape = utils.tensors.get_shape(loc_pred, 4)
    loc_pred = Reshape((shape[1]*shape[2]*num_anchors, 4))(loc_pred)

    # Class prediction.
    cls_pred = Conv2D(num_anchors * num_classes, [3, 3], padding='same', name='cls_{}'.format(layer))(base_layer)
    shape = utils.tensors.get_shape(cls_pred, 4)
    cls_pred = Reshape((shape[1]*shape[2]*num_anchors, num_classes))(cls_pred)

    return cls_pred, loc_pred


def deconvolutional_module(net, deconv_layer, ssd_layer):

    assert net[ssd_layer]._keras_shape[1] == 2 * net[deconv_layer]._keras_shape[1], 'deconv layer (HxWxD) needs to ' \
                                                                                    'be half the size of the ssd ' \
                                                                                    'layer (2Wx2HxD)'

    # deconvolution path
    x = Conv2DTranspose(512, [2, 2], padding='valid')(net[deconv_layer])
    x = Conv2D(512, [3, 3], padding='same')(x)
    x = BatchNormalization()(x)

    # ssd layer path
    y = Conv2D(512, [3, 3], padding='same')(net[ssd_layer])
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(512, [3, 3], padding='same')(y)
    y = BatchNormalization()(y)

    # merge and output
    print('x', x)
    print('y', y)
    z = Multiply()([x, y])
    print(z)
    net['dssd_{}'.format(ssd_layer)] = Activation('relu')(z)


def conv2d_block_padding(net, key, block, kernel_size, filters, kernel_stride):
    net['block{}_conv1'.format(block)] = Conv2D(filters[0],
                                                kernel_size[0],
                                                padding='same',
                                                name='block{}_conv1'.format(block))(net[key])
    net['block{}_bn1'.format(block)] = BatchNormalization()(net['block{}_conv1'.format(block)])
    net['block{}_pad1'.format(block)] = ZeroPadding2D((1, 1))(net['block{}_bn1'.format(block)])
    net['block{}_conv2'.format(block)] = Conv2D(filters[1],
                                                kernel_size[1],
                                                padding='valid',
                                                strides=kernel_stride,
                                                name='block{}_conv2'.format(block))(net['block{}_pad1'.format(block)])
    net['block{}_bn2'.format(block)] = BatchNormalization()(net['block{}_conv2'.format(block)])

    return 'block{}_conv2'.format(block)


def ssd_net(inputs,
            num_classes,
            feat_layers,
            anchor_sizes,
            anchor_ratios,
            normalizations,
            args):

    specs = ssd_layers_specs()
    net = dict()
    net['input'] = Input(shape=inputs)

    if args.pretrained is None:
        # Add base network
        vgg16 = keras.applications.vgg16.VGG16(input_tensor=net['input'],
                                               weights=None,
                                               pooling=True,
                                               include_top=False)
        for layer in feat_layers:
            if vgg16.get_layer(layer):
                net[layer] = vgg16.get_layer(layer).output

        # Add SSD layers
        net['block6'] = Conv2D(1024, (3, 3), padding='same', name='block6')(vgg16.output)
        net['block7'] = Conv2D(1024, (3, 3), padding='same', name='block7')(net['block6'])
        last = 'block7'
        for layer in range(8, 13):
            last = conv2d_block_padding(net, last, layer, **specs['block' + str(layer)])

    elif args.pretrained.startswith('block'):
        print('[INFO] transfer training')

        # use base model's weights
        base_model = keras.applications.vgg16.VGG16(input_tensor=net['input'],
                                                    weights='imagenet',
                                                    include_top=False)

        # and load them to this model
        vgg16 = keras.applications.vgg16.VGG16(input_tensor=net['input'],
                                               weights=None,
                                               include_top=False)

        # load only indicated weights
        for i, layer in enumerate(base_model.layers):
            weights = base_model.layers[i].get_weights()
            vgg16.layers[i].set_weights(weights)
            vgg16.layers[i].trainable = False
            print('loaded weights to :', layer.name)
            if args.pretrained in layer.name:
                break

        del base_model

        for layer in feat_layers:
            if vgg16.get_layer(layer):
                net[layer] = vgg16.get_layer(layer).output

        # Add SSD layers
        net['block6'] = Conv2D(1024, (3, 3), padding='same', name='block6')(vgg16.output)
        net['block7'] = Conv2D(1024, (3, 3), padding='same', name='block7')(net['block6'])
        last = 'block7'
        for layer in range(8, 13):
            last = conv2d_block_padding(net, last, layer, **specs['block' + str(layer)])

    elif os.path.exists(args.pretrained):

        # if using your own pretrained, give path to fmodel:
        #  'path/to/model/file/model.xx-x.xx.json'
        losses = utils.metrics.MultiboxLoss(args,
                                            negative_ratio=2.,
                                            negatives_for_hard=100,
                                            alpha=1.)
        model = utils.model.load_from_checkpoint(args.pretrained, losses)
        model.summary()

        return model

    else:

        raise ValueError('Invalid model. If default model is desired, set option to None')

    # # DSSD: dssd layers
    # dssd_layers = []
    # for i, layer in enumerate(reversed(feat_layers)):
    #     if i < 1:
    #         dssd_layers.append(layer)
    #     else:
    #         deconvolutional_module(net, dssd_layers[-1], layer)
    #         dssd_layers.append('dssd_{}'.format(layer))

    # Prediction and localisations layers.
    predictions = []
    logits = []
    localisations = []
    for i, layer in enumerate(feat_layers):
    # for i, layer in enumerate(dssd_layers):
        with tf.variable_scope(layer + '_box'):
            p, l = ssd_multibox_layer(net,
                                      layer,
                                      num_classes,
                                      anchor_sizes[i],
                                      anchor_ratios[i],
                                      normalizations[i])
        predictions.append(Activation('softmax')(p))
        logits.append(p)
        localisations.append(l)

    # concatenate all box proposals and its class predictions
    net['logits'] = Concatenate(axis=1,
                                name='conf')(logits)
    net['localisations0'] = Concatenate(axis=1,
                                        name='loc0')(localisations)

    # dummy layer: keras output and target shape match policy. See: https://github.com/keras-team/keras/issues/4781
    net['localisations'] = Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 1]])),
                                  name='loc')(net['localisations0'])

    net['predictions'] = Concatenate(axis=1,
                                     name='pred')(predictions)

    net['output'] = Concatenate(axis=2,
                                name='output')([net['predictions'],
                                                net['localisations'],
                                                net['logits']])
    model = Model(inputs=net['input'],
                  outputs=[net['logits'], net['localisations']], name='ssd_multibox')

    # multi-gpu training
    # n_gpu = len(get_available_gpus(-1))
    # model = multi_gpu_model(model, gpus=n_gpu)

    model.summary()

    return model


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))