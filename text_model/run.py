#!/usr/bin/python
# -*- coding:utf-8 -*-


import argparse

from text_models.model_skeleton import *


def run_train(args):
    options = \
        { "batch_size"  : ,
          "max_seq_len" : ,
          "vocab_file"  : args.vocab_file,
          "train_file"  : args.train_file,
          "need_evaluate" : True,
          "val_file"    : args.val_file,
          "max_epoch"   : ,
          "wembed_dim"  : ,
          "pre_trained_wembed" : args.embed_file,
          "finetune_wembed" : True,
          "train" : { "start_learning_rate" : 0.5,
                      "decay_ratio" : 0.5,
                      "decay_steps" : },
          "reg_type"    : "l2",
          "reg_scale"   : 1e-4,
          "grad_clip"   : 0.5,
        }

    train(options, args.save_dir, gpu_num=1)


def run_predict(args):
    predict(args.opt_file, args.model_path, args.pred_file, args.res_file)



if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="run type", dest="run_type")
    train_parser = subparsers.add_parser("train", help="run train process")
    pred_parser = subparsers.add_parser("predict", help="run predict process")

    train_parser.add_argument("--save_dir",   help="Directory for saving options and tf-models")
    train_parser.add_argument("--embed_file", help="Pre-trained word embedding file")
    train_parser.add_argument("--vocab_file", help="Vocabulary file")
    train_parser.add_argument("--train_file", help="Training set file")
    train_parser.add_argument("--val_file",   help="Validation set file")

    pred_parser.add_argument("--opt_file",   help="Json file of dumped options")
    pred_parser.add_argument("--model_path", help="Path of tf-model (prefix)")
    pred_parser.add_argument("--pred_file",  help="Input file for predict")
    pred_parser.add_argument("--out_file",   help="Output file for predict result")

    args = parser.parse_args()

    if args.run_type == "train":
        run_train(args)
    elif args.run_type == "predict":
        run_predict(args)
