import os
from argparse import Namespace
from evaluate import evaluate_function
from network import ssd_params_vars, eval_params_vars

import global_variables
global_variables.init()


# tuple-structures for parameters
EvalParams = eval_params_vars()
SSDParams = ssd_params_vars()

gt_file = os.path.join(global_variables.PICKLE_FOLDER, 'gt_test_fal_wbc_prep.pickle')
model_dir = os.path.join(global_variables.RESULTS_FOLDER, 'model_20XX_XX_XX_XX_XX_XX')
model_file = 'model.xx-0.00.json'
saveto = os.path.join(model_dir, 'results')
batch_size = 8
matching_thrs = 0.3

# evaluation params
eval_params = EvalParams(detection_threshold=0.0,  # give me back everything
                         nms_threshold=0.4,
                         select_top_k=400,
                         keep_top_k=200)

args = Namespace(gt=gt_file,
                 fmodel=model_file,
                 model_dir=model_dir,
                 saveto=saveto,
                 batch_size=batch_size,
                 matching_threshold=matching_thrs,
                 ssd_params=ssd_params_vars(),
                 eval_params=eval_params)

evaluate_function(args)
