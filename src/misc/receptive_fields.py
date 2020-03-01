# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center
# fall into padding)

import math

layer_names = ['conv1_1', 'conv1_2', 'maxpool1',
               'conv2_1', 'conv2_2', 'maxpool2',
               'conv3_1', 'conv3_2', 'conv3_3', 'maxpool3',
               'conv4_1', 'conv4_2', 'conv4_3', 'maxpool4',
               'conv5_1', 'conv5_2', 'conv5_3', 'maxpool5',
               'conv6',
               'conv7',
               'conv8_1',
               'conv8_2',
               'conv9_1',
               'conv9_2',
               'conv10_1',
               'conv10_2',
               'conv11_1',
               'conv11_2',
               'conv12_1',
               'conv12_2']

# [filter size, stride, padding]
convnet = [[3, 1, 1], [3, 1, 1], [2, 2, 0],
           [3, 1, 1], [3, 1, 1], [2, 2, 0],
           [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],#2
           [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],
           [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],#6
           [3, 1, 1],
           [1, 1, 0],
           [1, 1, 0],
           [3, 2, 1],
           [1, 1, 0],
           [3, 2, 1],
           [1, 1, 0],
           [3, 2, 1],
           [1, 1, 0],
           [3, 2, 1],
           [1, 1, 0],
           [4, 1, 1]]

imsize = 512


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    er_out = int(r_out * 0.68)
    start_out = start_in + ((k - 1) / 2 - pL) * j_in

    return n_out, j_out, r_out, er_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t Th. RF size: %s \n \t Eff. RF size: %s\t start: %s " % (
    layer[0], layer[1], layer[2], layer[3], layer[4]))


def onv2d_block_padding(net, key, block, skernels, nkernels, convstr):
    net['conv{}_1'.format(block)] = Conv2D(nkernels[0],
                                           skernels[0],
                                           padding='same',
                                           name='conv{}_1'.format(block))(net[key])
    net['bn{}_1'.format(block)] = BatchNormalization()(net['conv{}_1'.format(block)])
    net['pad{}'.format(block)] = ZeroPadding2D((1, 1))(net['bn{}_1'.format(block)])
    net['conv{}_2'.format(block)] = Conv2D(nkernels[1],
                                           skernels[1],
                                           padding='valid',
                                           strides=convstr,
                                           name='conv{}_2'.format(block))(net['pad{}'.format(block)])
    net['bn{}_2'.format(block)] = BatchNormalization()(net['conv{}_2'.format(block)])

    return 'bn{}_2'.format(block)

def ssd_layers_specs():

    return {'block6': {'filters': 1024,
                      'kernel_size': (3, 3),
                      'padding': 'same'},
            'block7': {'filters': 1024,
                      'kernel_size': (1, 1),
                      'padding': 'same'},
            'block8': {'skernels': [(1, 1), (3, 3)],
                      'nkernels': (256, 512),
                      'convstr': (2, 2)},
            'block9': {'skernels': [(1, 1), (3, 3)],
                      'nkernels': (128, 256),
                      'convstr': (2, 2)},
            'block10': {'skernels': [(1, 1), (3, 3)],
                       'nkernels': (128, 256),
                       'convstr': (2, 2)},
            'block11': {'skernels': [(1, 1), (3, 3)],
                       'nkernels': (128, 256),
                       'convstr': (2, 2)},
            'block12': {'skernels': [(1, 1), (4, 4)],
                       'nkernels': (128, 256),
                       'convstr': (1, 1)}
            }


layerInfos = []
if __name__ == '__main__':
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.68, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print("------------------------")


    # layer_name = raw_input("Layer name where the feature in: ")
    # layer_idx = layer_names.index(layer_name)
    # idx_x = int(raw_input("index of the feature in x dimension (from 0)"))
    # idx_y = int(raw_input("index of the feature in y dimension (from 0)"))
    #
    # n = layerInfos[layer_idx][0]
    # j = layerInfos[layer_idx][1]
    # r = layerInfos[layer_idx][2]
    # start = layerInfos[layer_idx][3]
    # assert (idx_x < n)
    # assert (idx_y < n)
    #
    # print("receptive field: (%s, %s)" % (r, r))
    # print("center: (%s, %s)" % (start + idx_x * j, start + idx_y * j))


    import keras
    from keras.models import Model
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
    import tensorflow as tf

    g = tf.Graph()
    specs = ssd_layers_specs()
    with g.as_default():
        net = dict()
        # net['input'] = Input(shape=(512, 512, 3))
        vgg16 = keras.applications.vgg16.VGG16(input_shape=(768, 768, 3),
                                               pooling=False,
                                               # pooling=True,
                                               weights=None,
                                               include_top=False)
        x = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='block5_pool')(vgg16.get_layer('block5_conv3').output)

        # define ssd layers
        def ssd_blocks(x, block, skernels, nkernels, convstr):
            x = Conv2D(nkernels[0], skernels[0], padding='same', name='block{}_conv1'.format(block))(x)
            # x = BatchNormalization()(x)
            if block != 12:
                x = ZeroPadding2D((1, 1))(x)
                x = Conv2D(nkernels[1], skernels[1], padding='valid', strides=convstr, name='block{}_conv2'.format(block))(x)
                # x = BatchNormalization()(x)

            return x

        x = Conv2D(name='block6', **specs['block6'])(vgg16.get_layer('block5_pool').output)
        x = Conv2D(name='block7', **specs['block7'])(x)
        x = ssd_blocks(x, 8, **specs['block8'])
        x = ssd_blocks(x, 9, **specs['block9'])
        x = ssd_blocks(x, 10, **specs['block10'])
        x = ssd_blocks(x, 11, **specs['block11'])
        x = ssd_blocks(x, 12, **specs['block12'])

        model = Model(inputs=vgg16.input,
                      outputs=x)

        # model.summary()
        for layer in model.layers:
            print(layer.name)
            if layer.name == 'input_1':
                continue
            model = Model(inputs=model.input,
                          outputs=layer.output)
            # model.summary()
            rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y =\
                tf.contrib.receptive_field.compute_receptive_field_from_graph_def(g.as_graph_def(),
                                                                                  model.input,
                                                                                  model.output)
            print(rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y)