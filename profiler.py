# python profiler: 0.5258905198867249 secs
import tensorflow as tf
import keras.backend as K

from collections import namedtuple
from PIL import Image, ImageOps
from scipy import stats
import numpy as np
import os
import pickle
from time import time

from evaluate import define_model_from_config
from network import SSDNet, eval_params_vars, ssd_params_vars
from utils.model import load_from_checkpoint
from utils.metrics import MultiboxLoss
from utils.tensors import get_shape
from tensorflow.contrib.distributions import percentile
import global_variables


def tf_contrast_stretching(image):
    tf_image = tf.to_float(image)

    values_range = tf.constant([0., 255.], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf_image, values_range, 256, dtype=tf.float32)

    condition = tf.less_equal(tf.argmax(histogram, output_type=tf.int32) - tf.argmin(histogram, output_type=tf.int32),
                              tf.constant(160, dtype=tf.int32))

    # histogram = tf.Print(histogram, [tf.shape(histogram)], message='histogram')

    pmin, pmax = tf.cond(condition,
                         lambda: [percentile(histogram, q=2, interpolation="nearest"),
                                  percentile(histogram, q=98, interpolation="nearest")],
                         lambda: [percentile(histogram, q=0, interpolation="nearest"),
                                  percentile(histogram, q=100, interpolation="nearest")])

    # remapping
    a = tf.constant(0., dtype=tf.float32)
    b = tf.constant(255., dtype=tf.float32)
    c = tf.cast(tf.reduce_min(tf.squeeze(tf.where(tf.equal(histogram, pmin)))), tf.float32)
    d = tf.cast(tf.reduce_min(tf.squeeze(tf.where(tf.equal(histogram, pmax)))), tf.float32)

    # c = tf.Print(c, [c], message='c')
    # d = tf.Print(d, [d], message='d')

    c_image = tf.cond(tf.equal(b - a, d - c),
                      lambda: tf_image,
                      lambda: (tf_image - c) * ((b - a) / (d - c)) + a)

    # c_image = tf.Print(c_image, [c_image], 'image')

    return c_image


def preprocessing(imgpath, input_shape=(512, 512)):
    # - read data
    # - contrast stretching
    # - scaling
    # - cropping

    def crop(im, output_shape):
        out_w, out_h = im.shape[:-1]
        in_w, in_h = output_shape

        sub_images = []
        for i in range(in_h // out_h):
            for j in range(in_w // out_w):
                sub_images.append(im[j * out_w, i * in_h, (j + 1) * out_w, (i + 1) * out_h, :])

        return sub_images

    # read image
    im = Image.open(imgpath)

    # stretch contrast
    stats.kurtosis(np.array(im).flatten())
    cutoff = 1. if np.array(im).max() - np.array(im).min() >= 160. else 0.
    contrast_img = ImageOps.autocontrast(im, cutoff=cutoff)

    # scale
    image = (contrast_img / 255.) * 2 - 1

    # crop them into sub-images
    images = crop(im, output_shape=input_shape)

    return images


def tf_preprocessing(image, input_shape):
    # - contrast stretching
    # - scaling
    # - cropping

    def crop(image, crop_size):
        in_h, in_w, in_d = get_shape(image, rank=3)

        # number of center crops per dimension
        n_width = tf.cast(tf.ceil(tf.divide(in_w, crop_size[1])), tf.int32)
        n_height = tf.cast(tf.ceil(tf.divide(in_h, crop_size[0])), tf.int32)

        # half crop size to compute all 4 corner coordinates
        half_w = tf.cast(tf.divide(crop_size[1], 2), tf.float32)
        half_h = tf.cast(tf.divide(crop_size[0], 2), tf.float32)

        # meshgrid of center coordinates
        X = tf.linspace(half_w, tf.cast(in_w, tf.float32) - half_w, num=n_width)
        Y = tf.linspace(half_h, tf.cast(in_h, tf.float32) - half_h, num=n_height)
        x, y = tf.meshgrid(X, Y)
        cx, cy = tf.reshape(x, [-1]), tf.reshape(y, [-1])

        # compute 4 corner coordinates
        y1, y2 = (cy - half_h) / tf.cast(in_h, tf.float32), (cy + half_h) / tf.cast(in_h, tf.float32)
        x1, x2 = (cx - half_w) / tf.cast(in_w, tf.float32), (cx + half_w) / tf.cast(in_w, tf.float32)
        box_coordinates = tf.cast(tf.stack([y1, x1, y2, x2], axis=1), tf.float32)

        # crop
        crops = tf.image.crop_and_resize(tf.expand_dims(image, axis=0),
                                         boxes=box_coordinates,
                                         box_ind=tf.zeros_like(x1, dtype=tf.int32),
                                         crop_size=crop_size)

        return crops, box_coordinates

    # load data
    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(queue)
    # image = tf.image.decode_jpeg(image_file, channels=3)

    # contrast
    cs_image = tf_contrast_stretching(image)

    # scale image
    s_image = tf.multiply(tf.div(cs_image, 255.), 2) - 1

    # crop into sub-images
    images, offsets = crop(s_image, crop_size=input_shape)

    return images, offsets


def tf_postprocessing(scr, loc, ssd_net, offsets):
    # - Decoding
    # - Sorting & top-k
    # - Translate coordinates to original position
    # - NMS (all together)

    # logits, localisations = y
    logits = scr
    localisations = loc

    # softmax to confidence scores
    predictions = K.softmax(logits)

    # decode, sort, select
    rscores, rbboxes = ssd_net.bboxes_detect([predictions, localisations])

    # concat batches into one sample
    def shift(boxes_tensor, offsets):
        boxes_tensor[:, 0] + offsets[0]
        boxes_tensor[:, 1] + offsets[1]
        boxes_tensor[:, 2] + offsets[2]
        boxes_tensor[:, 3] + offsets[3]

        return boxes_tensor, offsets

    # concat all together and shift coordinates.
    for c in rscores.keys():
        shifted_boxes, _ = tf.map_fn(lambda x: shift(x[0], x[1]),
                                     elems=[rbboxes[c], offsets],
                                     infer_shape=False,
                                     dtype=(tf.float32, tf.float32))
        boxes = tf.reshape(shifted_boxes, [-1, 4])
        scores = tf.reshape(rscores[c], [-1])

        indices = tf.image.non_max_suppression(boxes=boxes,
                                               scores=scores,
                                               max_output_size=int(ssd_net.eval.keep_top_k),
                                               iou_threshold=ssd_net.eval.nms_threshold)

        rbboxes[c] = tf.gather(boxes, indices)
        rscores[c] = tf.gather(scores, indices)

    return rscores, rbboxes


if __name__ == '__main__':

    global_variables.init()

    # define inference parameters
    InferParams = eval_params_vars()
    infer_params = InferParams(detection_threshold=0.5,  # give me back everything
                               nms_threshold=0.4,
                               select_top_k=400,
                               keep_top_k=200)

    # define script input arguments
    args = namedtuple('Profiler', ['model_dir',
                                   'fmodel',
                                   'cases',
                                   'ssd_params',
                                   'infer_params',
                                   'matching_threshold',
                                   'batch_size'])
    # set a value to them
    setattr(args, 'model_dir', os.path.join(global_variables.RESULTS_FOLDER, 'model_2018_3_26_10_55_34'))
    setattr(args, 'fmodel', 'model.111-1.44.json')
    setattr(args, 'cases', os.path.join(global_variables.PICKLE_FOLDER, 'gt_test_fal_wbc_prep.pickle'))
    setattr(args, 'ssd_params', ssd_params_vars())
    setattr(args, 'infer_params', infer_params)
    setattr(args, 'matching_threshold', 0.5) # just to make it work with load_from_checkpoint()
    setattr(args, 'batch_size', 4) # just to make it work with load_from_checkpoint()
    define_model_from_config(args, os.path.join(args.model_dir, 'model.config')) # define ssd_params

    K.set_learning_phase(0)
    # config = tf.ConfigProto()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=config)

    # sess = tf.InteractiveSession(config=config)
    K.set_session(sess)

    # define model
    ssd_net = SSDNet(args)
    # (just to make it work with load_from_checkpoint())
    losses = MultiboxLoss(args, negative_ratio=2., negatives_for_hard=100, alpha=1.)
    model = load_from_checkpoint(os.path.join(args.model_dir, 'checkpoints', args.fmodel), losses)

    # define input
    gt = pickle.load(open(args.cases, 'rb'), encoding='latin1')
    keys = [k for k in gt.keys()]
    filenames = [os.path.join(global_variables.IMAGES_PATH, k + '.jpg') for k in keys] #test_cases_filenames

    # define input queue
    # queue = tf.train.string_input_producer(filenames, num_epochs=None)
    image = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)

    # pipeline
    with tf.device('/GPU:0'):
        with K.get_session() as sess:
            # init
            sess.run(tf.global_variables_initializer())

            # profiler
            prof = tf.profiler.Profiler(sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # pipeline
            with tf.name_scope('preprocessing'):
                cropped_images, box_offsets = \
                    tf_preprocessing(image, input_shape=args.ssd_params.img_shape[:-1])
            with tf.name_scope('inference'):
                y_pred = model(cropped_images)
                # fn = K.function(model.inputs, model.outputs)#,
                #                 # options=run_options,
                #                 # run_metadata=run_metadata)
            with tf.name_scope('postprocessing'):
                rscores, rbboxes = \
                    tf_postprocessing(y_pred[0], y_pred[1], ssd_net, box_offsets)

            # execution
            pre, inf, post = 0, 0, 0
            for i, k in enumerate(filenames):

                if True:
                    kwargs = {'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                              'run_metadata': tf.RunMetadata()}
                # else:
                #     kwargs = {}
                #-----------------Read & Execute Entire Pipeline---------------
                image_arr = tf.keras.preprocessing.image.load_img(k)
                # scores, bboxes = sess.run([rscores, rbboxes],
                #                           feed_dict={image: image_arr},
                #                           run_metadata=run_metadata,
                #                           options=run_options)
                #--------------------------------------------------------------

                #-------------------------Pre-processing-----------------------
                s = time()
                crops, offsets = sess.run([cropped_images, box_offsets],
                                          feed_dict={image: image_arr},
                                          **kwargs)
                e = time()
                print('iteration {}, pre-processing, time elapsed: {} sec'.format(i, (e - s)))
                if i > 0:
                    pre += (e - s)
                # --------------------------------------------------------------

                # --------------------------Inference---------------------------
                # fn = K.function(model.inputs, model.outputs)
                                # **kwargs)
                s = time()
                y = sess.run(y_pred, feed_dict={cropped_images: crops})
                             # **kwargs)
                # y = fn([crops])
                e = time()
                print('iteration {}, inference, time elapsed: {} sec'.format(i, (e - s) / len(crops)))
                if i > 0:
                    inf += (e - s) / len(crops)
                # --------------------------------------------------------------

                # -----------------------Post-processing------------------------
                s = time()
                scores, bboxes = sess.run([rscores, rbboxes],
                                          feed_dict={y_pred[0]: y[0],
                                                     y_pred[1]: y[1],
                                                     box_offsets: offsets})
                                          # **kwargs)
                e = time()
                print('iteration {}, post-processing, time elapsed: {} sec'.format(i, (e - s)))
                if i > 0:
                    post += (e - s)
                # --------------------------------------------------------------

                # plt_bboxes_original(np.array(image_arr), np.arange(1, 3), bboxes, saveto='./testing.jpg')
                print('\n')

                # if i == 5:
                #     break
        print('Input Size: {}'.format(args.ssd_params.img_shape[0]))
        print('Crops length: {}'.format(len(crops)))
        print('pre time: ', (pre / (i + 1)) * 1000)
        print('inf time: ', (inf / (i + 1)) * 1000)
        print('post time: ', (post / (i + 1)) * 1000)
        print('[INFO] Inference Completed')

        # Profile the parameters of your model.
        opts = (tf.profiler.ProfileOptionBuilder()
                .select(['cpu_micros', 'accelerator_micros'])
                .with_stdout_output()
                # .with_timeline_output('timeline_pre.ctf.json')
                # .order_by(['micros'])
                .build())

        # prof.profile_name_scope(options=opts)

        # save performance profile
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open('{}.pre.timeline.ctf.json'.format(args.ssd_params.img_shape[0]), 'w') as f:
        #     f.write(trace.generate_chrome_trace_format())

        sess.close()
