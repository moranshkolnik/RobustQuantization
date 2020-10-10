#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
r"""
Here we implement the greedy search algorithm for automatic quantization.
"""
import torch
import torch.nn as nn
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, LinearQuantMode
from distiller.summary_graph import SummaryGraph
from distiller.model_transforms import fold_batch_norms
import distiller.modules
from distiller.data_loggers import collect_quant_stats
from distiller.models import create_model
from collections import OrderedDict, defaultdict
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os
import distiller.apputils as apputils
import re
import argparse
import numpy
from tqdm import tqdm
import pickle
import pdb
from distiller.quantization.clipped_linear import *

# __all__ = ['ptq_greedy_search']

msglogger = None


def get_default_args():
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--grid_resolution', '-gr', type=int, help='Number of intervals in the grid, one coordinate.',
                        default=11)
    parser.add_argument('--min_ratio', '-minr', type=float, help='min ration of the scale', default=0.7)
    parser.add_argument('--max_ratio', '-maxr', type=float, help='max ration of the scale', default=1.3)
    parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)
    parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
    parser.add_argument('--w_index', '-w_idx', nargs='+', type=int)
    args = parser.parse_args()
    return args


def override_odict(**kwargs):
    return OrderedDict(kwargs)


class MetaDataTmp:
    def __init__(self, num_bits_init, scale_ratio):
        self.num_bits = num_bits_init
        self.scale_ratio = scale_ratio


def fine_weight_tensor_by_name(model, name_in):
    for name, param in model.named_parameters():
        # print("name_in: " + str(name_in) + " name: " + str(name))
        if name == name_in:
            return param


def fine_module_by_name(model, name_in):
    for module_name, module in model.named_modules():
        # print("module_name: " + str(module_name))
        # print("norm_name: " + str(norm_name))
        q_weight_name = module_name + ".weight"
        if q_weight_name == name_in:
            return module


if __name__ == "__main__":
    print("START")
    args = get_default_args()
    # args.epochs = float('inf')  # hack for args parsing so there's no error in epochs
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    args = deepcopy(cc.args)  # Get back args after modifications in ClassifierCompressor.__init__
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)

    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)


    def test_fn(model):
        top1, top5, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        # pdb.set_trace()
        return top1, top5, losses


    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    args.device = next(model.parameters()).device
    if args.resumed_checkpoint_path:
        args.load_model_path = args.resumed_checkpoint_path
    if args.load_model_path:
        msglogger.info("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    dummy_input = torch.rand(*model.input_shape, device=args.device)

    meta_data_temp = MetaDataTmp(model.quantizer_metadata['params']['bits_weights'], 1.0)
    top1, top5, loss = test_fn(model)
    best_point = [1.0, loss, top1]
    print("best point: " + str(best_point))

    n = args.grid_resolution
    min_ratio = args.min_ratio
    max_ratio = args.max_ratio
    x = numpy.linspace(min_ratio, max_ratio, n)
    Z_loss = numpy.empty(n)
    Z_top1 = numpy.empty(n)

    # scale_ratio_range = numpy.arange(0.9, 1.1, 0.001)
    # res_dict = {}
    for i, x_ in enumerate(x):
        match_found = False
        total_bias_corr = 0
        for fp_name, param in model.named_parameters():
            if "float_" in fp_name:
                # if "module.layer2.0.conv1.float_weight" in fp_name:
                match_found = False
                norm_name = fp_name.replace("float_", "")
                for module_name, module in model.named_modules():
                    # print("module_name: " + str(module_name))
                    # print("norm_name: " + str(norm_name))
                    q_weight_name = module_name + ".weight"
                    if q_weight_name == norm_name:
                        match_found = True
                        # print("module_name: " + str(module_name) + ", shape: " + str(module.weight.shape))
                        # print("norm_name: " + str(norm_name) + ", shape: " + str(param.shape))
                        # pdb.set_trace()
                        # if x_ == 1.0:
                        # pdb.set_trace()
                        meta_data_temp = MetaDataTmp(model.quantizer_metadata['params']['bits_weights'], x_)
                        # # pdb.set_trace()
                        # # bias_corr = dorefa_quantize_param_v2(param,meta_data_temp)[1]
                        # # bias_corr = orig_q.mean()
                        #
                        # module.weight = dorefa_quantize_param_ptq(param, meta_data_temp).data

                        new_weight = dorefa_quantize_param_ptq(param, meta_data_temp).data

                        meta_data_temp = MetaDataTmp(model.quantizer_metadata['params']['bits_weights'], 1.0)
                        orig_weight = dorefa_quantize_param_ptq(param, meta_data_temp).data

                        if args.bcorr_w:
                            std_q = new_weight.view(new_weight.shape[0], -1).std(-1)
                            std_q = std_q.view(std_q.numel(), 1, 1, 1) if len(new_weight.shape) == 4 else std_q.view(
                                std_q.numel(), 1)

                            std_orig = orig_weight.view(orig_weight.shape[0], -1).std(-1)
                            std_orig = std_orig.view(std_orig.numel(), 1, 1, 1) if len(
                                orig_weight.shape) == 4 else std_orig.view(std_orig.numel(), 1)
                            bias_orig = orig_weight.view(orig_weight.shape[0], -1).mean(-1)
                            bias_orig = bias_orig.view(bias_orig.numel(), 1, 1, 1) if len(
                                orig_weight.shape) == 4 else bias_orig.view(bias_orig.numel(), 1)

                            std_corr_val = std_orig / std_q

                            # new_weight = new_weight*std_corr_val

                            bias_q = new_weight.view(new_weight.shape[0], -1).mean(-1)
                            bias_q = bias_q.view(bias_q.numel(), 1, 1, 1) if len(
                                new_weight.shape) == 4 else bias_q.view(bias_q.numel(), 1)

                            bias_corr_val = - bias_q + bias_orig
                            module.weight = new_weight + bias_corr_val

                            # module.weight = module.weight*std_corr_val

                            bias_corr_val_abs_mean = bias_corr_val.abs().mean()
                            total_bias_corr = total_bias_corr + bias_corr_val_abs_mean
                        else:
                            module.weight = new_weight

                if match_found == False:
                    print("norm_name: " + str(norm_name))
                    # dorefa_quantize_param()

        # TODO:: add activation scale

        top1, top5, loss = test_fn(model)
        Z_loss[i] = loss
        Z_top1[i] = top1
        str1 = "[x, loss, top1] = [{}, {}, {}]".format(x[i], Z_loss[i], Z_top1[i])
        print(str1)

    bcw_str = 'TT' if args.bcorr_w else 'FF'

    f_name = "{}_W{}A{}MINR{}_MAXR{}_GR{}_BCW{}.pkl".format(args.arch, 'ALL', None, str(min_ratio), str(max_ratio),
                                                            str(n), bcw_str)
    dir_fullname = os.path.join(os.getcwd(), args.experiment)
    if not os.path.exists(dir_fullname):
        os.makedirs(dir_fullname)
    f = open(os.path.join(dir_fullname, f_name), 'wb')
    data = {'X': x, 'Z_loss': Z_loss, 'Z_top1': Z_top1, 'best_point': best_point}
    pickle.dump(data, f)
    f.close()
    print("Data saved to {}".format(f_name))
