## Customize convert_tf_checkpoint_to_pytorch.py
## only need the bert folder path

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from modeling import BertConfig, BertForPreTraining

def convert_tf_checkpoint_to_pytorch(bert_model_path):

    tf_checkpoint_path = os.path.join(bert_model_path,'bert_model.ckpt')
    bert_config_file   = os.path.join(bert_model_path,'bert_config.json')
    pytorch_dump_path  = os.path.join(bert_model_path,'pytorch_model.bin')

    
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if name[-1] in ["adam_v", "adam_m"]:
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("\nSave PyTorch model to {}".format(pytorch_dump_path))
    print('Convert TF -> Pytorch Done!!\n')
    torch.save(model.state_dict(), pytorch_dump_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the pretrained bert model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.bert_model_path)
    
