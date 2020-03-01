import numpy as np
import os, pickle
from keras.preprocessing import image
import argparse
import matplotlib.pyplot as plt
from numpy import random

from train import model_settings
import global_variables
from utils import to_relative_coordinates
from data import preprocessing as preproc
from utils.visualization import plt_bboxes


def generator(gt, keys, args, ssd, mode='', transform=False, random=True):
    """
        Utility function to generate input data in batches

    :param gt: dict containing gt bounding boxes per image, {image key: [xmin, xmax, ymin, ymax, -hot encoded labels-]}.
    :param keys: list of image keys (absolute path of image).
    :param args: training script arguments relevant to the training process

    :return: tuple of numpy arrays (image, target)
    """
    # list of anchor boxes
    ssd_anchors = ssd.anchors()

    while True:
        if random:
            np.random.shuffle(keys)
        targets, inputs = [], []

        for k in keys:
            if os.path.exists(os.path.join(global_variables.PREPROC_FOLDER, k)):

                # loading data
                img = np.array(image.load_img(os.path.join(global_variables.PREPROC_FOLDER, k))) / 255.
                bb = gt[k][0]
                objclass = gt[k][1]

                assert len(bb) == len(objclass), 'something is up'

                if transform:
                    # data augmentation
                    # if np.random.rand() > 0.5:
                    #     img, bb = clipped_zoom_in(img, bb, zoom_factor=1.3)

                    if np.random.rand() > 2:
                        print('rotate')
                        rot_angle = np.random.choice([90, 180, 270])
                        img, bb = preproc.rotate(img, bb, rot_angle)

                    if np.random.rand() > -1:
                        print('flip')
                        flip_axis = np.random.choice([0, 1])
                        img, bb = preproc.flip(img, bb, flip_axis)

                # check for negative images
                if bb.any():
                    bb = to_relative_coordinates(bb, (512, 512, 3))
                    objclass = np.int64(np.argmax(objclass, axis=1) + 1)
                else:
                    bb = np.zeros((0, 4), dtype='float32')
                    objclass = np.zeros((0,), dtype='int64')

                # encode ground truth labels and bboxes.
                switcher = {'tf': '', #ssd.bboxes_encode(objclass, bb, ssd_anchors)
                            'np': ssd.bboxes_encode_numpy(objclass, bb, ssd_anchors)}
                classes, localisation, scores = switcher.get(mode, IOError('tf or np'))

                # append to list
                inputs.append(img)

                targets.append(np.hstack([classes, localisation, scores]))

                if len(targets) == args.batch_size:
                    yield np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)
                    inputs, targets = [], []


def prepare_input_target_format_for_detection(target, args):
    shape = target.shape

    label = target[:, :, 0]
    loc   = target[:, :, 1:-1]
    score = target[:, :, -1]

    one_hot_labels = np.zeros((shape[0], shape[1], args.ssd_params.num_classes), dtype='float32')
    for i, l in enumerate(label):
        one_hot_labels[i] = np.eye(3)[np.int32(l)]

    logits = one_hot_labels * score[..., None]

    return np.concatenate([one_hot_labels, loc, logits], axis=-1)


def test_script(args):
    # vars
    mode = 'np'
    colors = {1: (random.random(), random.random(), random.random()),
              2: (random.random(), random.random(), random.random())}
    gt = pickle.load(open(args.gt, 'rb'), encoding='latin1')
    ssd_net, model = model_settings(args)

    # test
    gen = generator(gt, list(gt.keys()), args, ssd_net, mode=mode, transform=True, random=True)
    for j, sample in enumerate(gen):
        inp, out = sample
        out = prepare_input_target_format_for_detection(out, args)

        switcher = {'tf': '', #ssd_net.bboxes_detect(out),
                    'np': ssd_net.bboxes_detect_numpy(out)}
        classes, scores, bboxes = switcher.get(mode, IOError('tf or np'))
        print('1')
        for i, img in enumerate(inp):
            print(img.shape)
            fig, ax = plt.subplots(1, 1)
            print('4')
            for cls in range(1, 3):
                print('5')
                idx = np.where(classes[i] == cls)[0]
                sc = scores[i][idx]
                bb = bboxes[i][idx]
                print('6')
                plt_bboxes(img, cls, sc, bb, colors, ax, linewidth=1.5)
                print('7')
            fig.savefig('./tests/data_gen_flip_{}_{}.jpg'.format(i, j))
            print('2')

        if j == 2:
            break
        print('3')
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate SSD Multibox to detect Malaria.')

    parser.add_argument('--gt', dest='gt', help='ground truth pickled file')
    parser.add_argument('--output_dir', dest='saveto', help='path to save results')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=8, help='batch size')
    parser.add_argument('--ssd_params', dest='ssd_params', help='SSD multibox parameters')

    args = parser.parse_args()

    if not args.gt:
        parser.error('gt file required')

    global_variables.init()
    test_script(args)