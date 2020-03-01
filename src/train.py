import os, sys, io
import pickle
import random
import argparse
import numpy as np
import datetime, pytz

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

from utils.metrics import MultiboxLoss
from utils.data import preprocessing as preproc
from utils.model import to_relative_coordinates
from utils.visualization import plot_anchors_on_an_image
from network import SSDNet, ssd_feat_shapes_from_net, ssd_size_bounds_to_values, ssd_params_vars

import global_variables


class Unbuffered:
    '''
        Redirects std.out to a log file
    '''
    def __init__(self, stream, f):
        self.stream = stream
        self.file = f

    def flush(self):
        pass

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)


class SaveModelCallback(Callback):
    """
        Checkpointer. Using keras checkpoint callback gives different/incorrect results than using
        a custom one with different saving/loading fucntions.
    """
    def __init__(self, path):
        self.saveto = path

    def on_train_begin(self, logs=None):
        if not os.path.exists(self.saveto):
            os.makedirs(self.saveto)

    def on_epoch_end(self, epoch, logs={}):

        fmodel = os.path.join(self.saveto, 'model.{:02d}-{:.2f}.json'.format(epoch, logs['val_loss']))
        fweights = os.path.join(self.saveto, 'weights.{:02d}-{:.2f}.hdf5'.format(epoch, logs['val_loss']))

        model_json = self.model.to_json()
        with open(fmodel, 'w') as json_file:
            json_file.write(model_json)

        self.model.save_weights(fweights)

        print('model saved to disk!')


def summary_data(keys, gt):
    """
        Returns a quantitative summary of the dataset.

    :param keys: (list) keys from gt dict that represent image file names.
    :param gt: (dict) ground truth dictionary {'filename': (bboxes, labels)}

    :return: count_images: (dict) 'positive' and 'negative' image counts {'label':count}
             count_labels: (dict) total count per label {'label':count}
    """
    # prepare dictionaries
    count_labels, name_labels = {}, {}
    for key, value in global_variables.CLASSES.items():
        if key != 'background':
            count_labels[key] = 0
            name_labels[value] = key

    # register the frequency and type of objects in the image
    count_images = {'positives': 0, 'negatives': 0}
    for k in keys:
        objclass = gt[k][1]
        if objclass.any():
            for c in objclass:
                label = name_labels[np.int64(np.argmax(c)+1)]
                count_labels[label] += 1
            count_images['positives'] += 1
        else:
            count_images['negatives'] += 1

    return count_images, count_labels


def write_summary_to_file(train, val, gt, out_file):
    """
        Writes data summary to an output file.

    :param train: (list) image filenames from the training set
    :param val: (list) image filenames from the validation set
    :param gt: (dict) ground truth dictionary {'filename': (bboxes, labels)}
    :param out_file: (string) path to summary file

    :return: None
    """

    with open(out_file, "w") as text_file:
        for name, dataset in zip(['TRAINING SET', 'VALIDATION SET'], [train, val]):
            text_file.write("\n--------{}------------- \n".format(name))
            count_images, count_labels = summary_data(dataset, gt)
            text_file.writelines("Num of Images: {} \n".format(len(dataset)))

            text_file.writelines('\n')
            text_file.writelines('image summary \n')
            for key in count_images:
                text_file.writelines("{}: {}\n".format(key, count_images[key]))

            text_file.writelines('\n')
            text_file.writelines('labels summary \n')
            for key in count_labels:
                text_file.writelines("{}: {}\n".format(key, count_labels[key]))


def train_test_split(args, train_ratio):
    """
        Train/test split according to train ratio.

    :param args: (namedtuple) training args passed to script
    :param train_ratio: (float) ratio of data points in gt to assign to the training set

    :return: gt: (dict) ground truth dictionary {'filename': (bboxes, labels)},
             x_train (list) of image filenames from training set,
             x_val (list) of image filenames from validation set,
             num_train (int) number of training samples,
             num_val (int) number of validation samples
    """
    gt = pickle.load(open(args.gt, 'rb'), encoding='latin1')

    keys = sorted(gt.keys(), key=lambda k: random.random())
    num_train = int(np.ceil(len(keys) * train_ratio))
    x_train = keys[:num_train]
    x_val = keys[num_train:]

    num_train = int(np.ceil(len(x_train) / args.batch_size))
    num_val = int(np.ceil(len(x_val) / args.batch_size))

    return gt, x_train, x_val, num_train, num_val


def save_train_test_sets(train, test, folder):
    """
        Serialize (pickle) train/val sets key names.

    :param train: (list) of image filenames
    :param test: (list) of image filenames
    :param folder: (string) saveto path

    :return: None
    """
    pickle.dump(train, open(os.path.join(folder, "train_set.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(folder, "test_set.pkl"), "wb"))


def get_optimizer(args):
    """
        Choose optimizer
    :param args: (namedtuple) training args passed to script

    :return: optimizer (keras obj)
    """
    if args.optimizer == 'adam':
        optimizer = Adam(lr=args.lr)

    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=args.lr)

    elif args.optimizer == 'sgd':
        optimizer = SGD(lr=args.lr, momentum=0.9, nesterov=True)

    return optimizer


def generator(gt, keys, args, ssd, random=True):
    """
        Utility function to generate input data in batches

    :param gt: (dict) containing gt bounding boxes per image, {image key: [xmin, xmax, ymin, ymax, -hot encoded labels-]}.
    :param keys: (list) of image keys (absolute path of image).
    :param args: (namedtuple) training script arguments relevant to the training process
    :param ssd: (object) SSD model class object

    :return: (tuple) containing (image, target),
              where image is a (numpy array) and target is a (dict) with (numpy arrays) values
    """
    # list of anchor boxes
    ssd_anchors = ssd.anchors()

    while True:
        if random:
            np.random.shuffle(keys)

        loc_targets, conf_targets, inputs = [], [], []

        for k in keys:
            if os.path.exists(os.path.join(global_variables.IMAGES_PATH, k + '.jpg')):

                # loading data
                img = (np.array(image.load_img(os.path.join(global_variables.IMAGES_PATH, k + '.jpg'))) / 255.) * 2 - 1
                bb = gt[k][0]
                objclass = gt[k][1]

                assert len(bb) == len(objclass), 'something is up'

                # data augmentation
                # if np.random.rand() > 0.5:
                #     img, bb = clipped_zoom_in(img, bb, zoom_factor=1.3)

                if np.random.rand() > 0.5:
                    rot_angle = np.random.choice([90, 180, 270])
                    img, bb = preproc.rotate(img, bb, rot_angle)

                if np.random.rand() > 0.5:
                    flip_axis = np.random.choice([0, 1])
                    img, bb = preproc.flip(img, bb, flip_axis)

                # check for negative images
                if bb.any():
                    bb = to_relative_coordinates(bb, global_variables.INPUT_SHAPE)
                    objclass = np.int64(np.argmax(objclass, axis=1) + 1)
                else:
                    bb = np.zeros((0, 4), dtype='float32')
                    objclass = np.zeros((0,), dtype='int64')

                # encode ground truth labels and bboxes.
                classes, localisation, scores = ssd.bboxes_encode_numpy(objclass, bb, ssd_anchors)

                # append to list
                inputs.append(img)
                loc_targets.append(np.hstack([localisation, scores]))
                conf_targets.append(np.hstack([classes, classes, scores]))

                if len(loc_targets) == args.batch_size:
                    yield np.array(inputs, dtype=np.float32), \
                          {'conf': np.array(conf_targets, dtype=np.float32),
                           'loc': np.array(loc_targets, dtype=np.float32)}

                    inputs, loc_targets, conf_targets = [], [], []


def define_output_files(args, x_train, x_val, gt, write=True):
    """
        Create dirs and, config and summary files.

    :param args: (namedtuple) training script arguments relevant to the training process
    :param x_train: train: (list) of train image filenames
    :param x_val: (list) of val image filenames
    :param gt: (dict) ground truth dictionary {'filename': (bboxes, labels)}
    :param write: (boolean) if to write data split to disk and data summary to file

    :return: (tuple of strings) directory names
    """
    ams_tz = pytz.timezone('Europe/Amsterdam')
    model_dir = os.path.join(global_variables.RESULTS_FOLDER,
                             'model_' +
                             str(datetime.datetime.now(ams_tz).year) + '_' +
                             str(datetime.datetime.now(ams_tz).month) + '_' +
                             str(datetime.datetime.now(ams_tz).day) + '_' +
                             str(datetime.datetime.now(ams_tz).hour) + '_' +
                             str(datetime.datetime.now(ams_tz).minute) + '_' +
                             str(datetime.datetime.now(ams_tz).second) + '_' +
                             str(np.random.choice(99)))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_config    = os.path.join(model_dir, 'model.config')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    tfboard_dir     = os.path.join(model_dir, 'tfboard')
    summary_file    = os.path.join(model_dir, 'summary.txt')

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(tfboard_dir):
        os.makedirs(tfboard_dir)

    # write model config file
    with open(model_config, 'w') as f:
        # write first general params
        for key, value in vars(args).items():
            if key in ['ssd_params', 'eval_params']:
                continue
            f.write(key + ': ' + str(value) + '\n')

        # at the end, write ssd and eval params
        for key, value in vars(args).items():
            if key in ['ssd_params', 'eval_params']:
                f.write('\t\t\t\t\t\t\t' + key + '\n')
                for p in value._fields:
                    f.write(p + ':' + str(getattr(value, p)) + '\n')
                f.write('\n')
            else:
                continue

    # redirect stdout to training.log
    sys.stdout = Unbuffered(sys.stdout, open(os.path.join(model_dir, 'training.log'), 'w'))

    # write train/val data keys for reproduction
    if write:
        write_summary_to_file(x_train, x_val, gt, summary_file)
        save_train_test_sets(x_train, x_val, model_dir)

    return model_dir, checkpoints_dir, tfboard_dir


def model_settings(args):
    """
        Instantiates model according to settings in args.

    :param args: (namedtuple) training script arguments relevant to the training process

    :return: ssd_net (obj) SSDNet class object
             model (obj) keras model object
    """
    # instantiate model class
    ssd_net = SSDNet(args)

    # compute anchor sizes per ssd layer (if set manually, uncomment this line)
    ssd_net.params.anchor_sizes.extend(ssd_size_bounds_to_values(ssd_net.params.anchor_size_bounds,
                                                                 len(ssd_net.params.feat_layers),
                                                                 img_shape=ssd_net.params.img_shape))

    # instantiate keras model
    model = ssd_net.net()

    # compute feature shapes sizes
    ssd_net.params.feat_shapes.extend(ssd_feat_shapes_from_net(model, 'loc_'))
    print('feat_shapes', ssd_net.params.feat_shapes)

    return ssd_net, model


def train_function(args):
    """
        Training pipeline.

    :param args: (namedtuple) training script arguments relevant to the training process


    :return: None
    """
    print('HERE', isinstance(sys.stdin, io.TextIOWrapper))
    exit()

    with tf.Session() as sess:

        K.set_session(sess)
        K.set_learning_phase(1)

        # get train/val sets
        gt, x_train, x_val, num_train, num_val = train_test_split(args, 0.95)

        # set params of SSD network and get an instance of the model class and the keras network
        ssd_net, model = model_settings(args)

        # # plot anchors to visually assess bboxes size.
        # for i, k in enumerate(gt.keys()):
        #     img = np.array(image.load_img(os.path.join(global_variables.IMAGES_PATH, k + '.jpg')))
        #     plot_anchors_on_an_image(img, ssd_net, './', 'anchors_' + str(i))
        #     if i == 5:
        #         break
        # exit()

        # define model dirs and files
        model_dir, checkpoints_dir, tfboard_dir = define_output_files(args, x_train, x_val, gt, write=True)
        print('Model directory: {}'.format(str(model_dir)))

        # define loss
        losses = MultiboxLoss(args, negative_ratio=2., negatives_for_hard=100, alpha=1.)

        # define callbacks
        # def scheduler(epoch):
        #     drop = 0.5
        #     epochs_drop = 15.0
        #     # return args.lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))

        callbacks = [SaveModelCallback(os.path.join(checkpoints_dir)),
                     # LearningRateScheduler(scheduler),
                     # EarlyStopping(monitor='val_loss', patience=15),
                    # keras.callbacks.TensorBoard(log_dir=tfboard_dir,
                    #                              histogram_freq=1,
                    #                              write_graph=True,
                    #                              write_images=True),
                     ]

        # compile model
        model.compile(optimizer=args.optimizer, loss={'conf': losses.confidence_loss,
                                                      'loc': losses.localisation_loss})

        # train
        history = model.fit_generator(generator(gt,
                                                x_train,
                                                args,
                                                ssd_net,
                                                random=True),
                                      steps_per_epoch=num_train,
                                      epochs=args.epochs,
                                      verbose=1,
                                      validation_data=generator(gt,
                                                                x_val,
                                                                args,
                                                                ssd_net,
                                                                random=True),
                                      validation_steps=num_val,
                                      callbacks=callbacks)

        # save history variable
        pickle.dump(history.history, open(os.path.join(model_dir, 'train_history.pkl'), 'wb'))
        print('train history saved!')


if __name__ == '__main__':

    global_variables.init()
    SSDParams = ssd_params_vars()

    parser = argparse.ArgumentParser(description='Train SSD Multibox to detect malaria.')

    parser.add_argument('--gt', dest='gt', help='ground truth pickled file')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=8, help='batch size')
    parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='num. of epochs')
    parser.add_argument('--learning_rate', '--lr', type=float, dest='lr', default=.0001, help='learning rate')
    parser.add_argument('--matching_thrs', type=float, dest='matching_threshold', default=.5, help='jaccard index ')
    parser.add_argument('--optimizer', '--opt', type=str, dest='optimizer', default='adam', help='optimizer')
    parser.add_argument('--comment', type=str, dest='comment', help='comment to display in training log')
    parser.add_argument('--ssd_params', dest='ssd_params', default=None, help='SSD multibox parameters')
    parser.add_argument('--warm_start', dest='pretrained', type=str, default=None,
                        help='Warm start: pretrained W to init the network with')

    #-----Uncomment encapsuled sections below if execution from command line is desired/required-----------

    if isinstance(sys.stdin, io.TextIOWrapper):
        #----------------------------------------- pt. 1------------------------------------------------------
        parser.add_argument('--img_shape', dest='img_shape', nargs='*', type=int)
        parser.add_argument('--num_classes', dest='num_classes', type=int)
        parser.add_argument('--no_annotation_label', dest='no_annotation_label', type=float)
        parser.add_argument('--feat_layers', dest='feat_layers', nargs='*', type=str)
        parser.add_argument('--anchor_size_bounds', dest='anchor_size_bounds', nargs='*', type=float)
        parser.add_argument('--anchor_ratios', dest='anchor_ratios', nargs='*', type=list)
        parser.add_argument('--anchor_sizes', dest='anchor_sizes', nargs='*')
        parser.add_argument('--anchor_steps', dest='anchor_steps', nargs='*', type=int)
        parser.add_argument('--anchor_offset', dest='anchor_offset', type=float)
        parser.add_argument('--normalizations', dest='normalizations', nargs='*', type=int)
        parser.add_argument('--prior_scaling', dest='prior_scaling', nargs='*', type=float)
        parser.add_argument('--img_dir', dest='img_dir', type=str)
        # ----------------------------------------- pt. 1------------------------------------------------------

    args = parser.parse_args()

    if not args.gt:
        parser.error('gt file required')

    supported = ['adam', 'rmsprop', 'sgd']
    if args.optimizer not in supported:
        parser.error('Supported optimizers include: {}'.format(supported))

    # ----------------------------------------- pt. 3 ----------------------------------------------------------
    if isinstance(sys.stdin, io.TextIOWrapper):
        if args.anchor_sizes:
            if len(args.anchor_sizes) < len(args.feat_layers) * 2:
                parser.error('Min. and max. sizes need to be specified per layer. '
                             'Max values can be "None". '
                             'Min. values are floats.')

    # cast them into float if not 'None'
    # args.anchor_sizes = [None if i == 'None' else float(i) for i in args.anchor_sizes]

    # make a list of tuples (min, max)
    # args.anchor_sizes = [tuple(args.anchor_sizes[2 * i: 2 * (i + 1)]) for i in range(int(len(args.anchor_sizes) / 2))]
    # ----------------------------------------- pt. 3 ----------------------------------------------------------

    ssd_params = SSDParams(img_shape=args.img_shape,
                           num_classes=args.num_classes,
                           no_annotation_label=args.no_annotation_label,
                           feat_layers=args.feat_layers,
                           feat_shapes=[],
                           anchor_size_bounds=args.anchor_size_bounds,
                           anchor_sizes=args.anchor_sizes,
                           anchor_ratios=args.anchor_ratios,
                           anchor_steps=args.anchor_steps,
                           anchor_offset=args.anchor_offset,
                           normalizations=args.normalizations,
                           prior_scaling=args.prior_scaling,
                           matching_threshold=args.matching_threshold)


    # ----------------------------------------- pt. 2 ----------------------------------------------------------

    if isinstance(sys.stdin, io.TextIOWrapper):
        args.ssd_params = ssd_params
        global_variables.IMAGES_PATH = os.path.join(global_variables.DATA_PATH, 'images', args.img_dir)
        global_variables.INPUT_SHAPE = tuple(args.img_shape)
        args.gt = os.path.join(global_variables.PICKLE_FOLDER, args.gt)

        if args.pretrained is not None:
            if '/' in args.pretrained:
                model_dir, model = os.path.split(args.pretrained)
                args.pretrained = os.path.join(global_variables.RESULTS_FOLDER, model_dir, 'checkpoints', model)
    # ----------------------------------------- pt.2 -----------------------------------------------------------

    train_function(args)
