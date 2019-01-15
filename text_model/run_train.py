#!/usr/bin/python
# -*- coding:utf-8 -*-


import argparse

from text_models.model_skeleton import *


def main(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    options = \
        { 'batch_size'  : ,
          'max_seq_len' : ,
          'vocab_file'  : args.vocab_file,
          'train_file'  : args.train_file,
          'need_evaluate' : True,
          'val_file'    : args.val_file,
          'max_epoch'   : ,
          'wembed_dim'  : ,
          'pre_trained_wembed' : args.embed_file,
          'finetune_wembed' : True,
          'train' : { 'start_learning_rate' : 0.5,
                      'decay_ratio' : 0.5,
                      'decay_steps' : },
          'reg_type'    : 'l2',
          'reg_scale'   : 1e-4,
          'grad_clip'   : 0.5,
        }

    train(options, args.save_dir, gpu_num=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',   help='Directory for saving options and tf-models')
    parser.add_argument('--embed_file', help='Pre-trained word embedding file')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_file', help='Training set file')
    parser.add_argument('--val_file',   help='Validation set file')

    args = parser.parse_args()
    main(args)

