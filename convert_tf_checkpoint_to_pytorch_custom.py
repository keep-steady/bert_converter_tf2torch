# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
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
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re, os
import argparse
import tensorflow as tf
import torch
import numpy as np

from modeling import BertConfig, BertModel

def convert(bert_model_path):
    
    tf_checkpoint_path = os.path.join(bert_model_path,'bert_model.ckpt')
    bert_config_file   = os.path.join(bert_model_path,'bert_config.json')
    pytorch_dump_path  = os.path.join(bert_model_path,'pytorch_model.bin')
    
    
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    model = BertModel(config)

    # Load weights from TF model
    path = tf_checkpoint_path
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split('/')
        if name[0] in ['redictions', 'eq_relationship']:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] in set(['adam_m', 'adam_v', 'l_step']):
                continue
            if l[0] == 'kernel':
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
        # try:
        #     assert pointer.shape == array.shape
        # except AssertionError as e:
        #     print(pointer)
        #     e.args += (pointer.shape, array.shape)
        #     raise
        # print(array)
        print(name)
        if name[0] == 'l_step':
            continue
        # if name[0] == 'kernel':
        #     array = np.transpose(array, (1, 0))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), pytorch_dump_path)
    # Save pytorch-model
    print("\nSave PyTorch model to {}".format(pytorch_dump_path))
    print('Convert TF -> Pytorch Done!!\n')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the pretrained bert model.")
    args = parser.parse_args()
    convert(args.bert_model_path)
    
    
    
