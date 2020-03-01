import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import namedtuple

import keras.backend as K
import tensorflow as tf
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_activation
from vis.visualization import get_num_filters
from vis.input_modifiers import Jitter

import global_variables
from evaluate import define_model_from_config
from network import SSDNet, eval_params_vars, ssd_params_vars
from utils.model import load_from_checkpoint
from utils.metrics import MultiboxLoss

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
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

# define model
ssd_net = SSDNet(args)
# (just to make it work with load_from_checkpoint())
losses = MultiboxLoss(args, negative_ratio=2., negatives_for_hard=100, alpha=1.)
model = load_from_checkpoint(os.path.join(args.model_dir, 'checkpoints', args.fmodel), losses)

# filters
selected_indices = []
for layer_name in ['block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']:
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    selected_indices.append(filters)

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = visualize_activation(model, layer_idx,
                                   filter_indices=idx,
                                   input_modifiers=[Jitter(0.05)])

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(idx))
        vis_images.append(img)

    # Generate stitched image palette with 5 cols so we get 2 rows.
    stitched = utils.stitch_images(vis_images, cols=5)
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()