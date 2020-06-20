import os
from argparse import Namespace
from train import train_function
from network import ssd_params_vars, eval_params_vars

import global_variables
global_variables.init()


# tuple-structures for parameters
EvalParams = eval_params_vars()
SSDParams = ssd_params_vars()


# Settings
gt_file       = os.path.join(global_variables.PICKLE_FOLDER, 'gt_train2_fal_wbc_prep.pickle')
epochs        = 2
batch_size    = 16
optimizer     = 'adam'
learning_rate = 0.001
matching_thrs = 0.5
comment       = ''
pretrained    = None
# pretrained    = os.path.join(global_variables.RESULTS_FOLDER,
#                              'model_2018_1_22_14_50',
#                              'checkpoints',
#                              'model.17-1.68.json')

# network params
ssd_params = SSDParams(img_shape=global_variables.INPUT_SHAPE,
                       num_classes=global_variables.NUM_CLASSES,
                       no_annotation_label=0.,
                       feat_layers=['block4_conv1', 'block5_conv1', 'block6'],
                       # feat_layers=['block4_conv1',
                       #              'block6',
                       #              'block8_conv2',
                       #              'block9_conv2',
                       #              'block10_conv2',
                       #              'block11_conv2',
                       #              'block12_conv2'
                       #              ],
                       feat_shapes=[],  # defined in train_function() when model is created
                       anchor_size_bounds=[0.08, 0.54],  #512
                       anchor_sizes=[],
                       # anchor_sizes=[(40., 60., 70.), (100., 130., 160.), (190., 210., 240.)],
                       # anchor_sizes=[(40.96, 71.68),
                       #               (71.68, 102.4),
                       #               (102.4, 133.12),
                       #               (133.12, 163.84),
                       #               (163.84, 194.56),
                       #               (194.56, 225.28),
                       #               (225.28, 256.0),
                       #               (256.0, 286.72)
                       #               ],
                       anchor_ratios=[(1, 1, 1),
                                      (1, 1, 1),
                                      (1, 1, 1),
                                      # (1, 1),
                                      # (1, 1),
                                      # (1, 1),
                                      # (1, 1),
                                      # (1, 1)
                                      ],
                       anchor_steps=[8, 16, 32, 64, 128, 256, 512],  # for now these are dummy values
                       anchor_offset=0.5,  # in relation to anchor steps per detection layer (margin)
                       normalizations=[20, -1, -1, -1, -1, -1, -1],
                       prior_scaling=[0.1, 0.1, 0.2, 0.2],
                       matching_threshold=matching_thrs)

# add them to input args
args = Namespace(gt=gt_file,
                 batch_size=batch_size,
                 epochs=epochs,
                 lr=learning_rate,
                 matching_threshold=matching_thrs,
                 optimizer=optimizer,
                 comment=comment,
                 ssd_params=ssd_params,
                 pretrained=pretrained)

train_function(args)
