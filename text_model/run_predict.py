#!/usr/bin/python
# -*- coding:utf-8 -*-


import argparse

from text_models.model_skeleton import *


def main(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    predict(args.opt_file, args.model_path, args.pred_file, args.res_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_file',   help='Json file of dumped options')
    parser.add_argument('--model_path', help='Path of tf-model (prefix)')
    parser.add_argument('--pred_file',  help='Input file for predict')
    parser.add_argument('--out_file',   help='Output file for predict result')

    args = parser.parse_args()
    main(args)

