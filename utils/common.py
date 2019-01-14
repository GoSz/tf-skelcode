#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np


TF_DTYPE = tf.float32
TF_DTYPE_INT = tf.int32
NP_DTYPE = np.float32
NP_DTYPE_INT = np.int32


def dump_options(options, out_path):
    import json
    with open(out_path, 'w') as opt_f:
        json.dump(options, opt_f)

def load_options(opt_file):
    import json
    with open(opt_file) as opt_f:
        return json.load(opt_f)


def count_line(filename):
    with open(filename, 'r') as f:
        i = 0
        for line in f:
            i += 1
    return i


def feedable_iterator(*args, **kwargs):
    handle   = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
            handle, output_types=kwargs["dtype"],
            output_shapes=kwargs["shape"])
    return iterator


def get_regularizer(reg_type, *reg_scale):
    """
    Get regularizer for `tf.get_variable`.

    Args:
        reg_type: type of regularizer.
        reg_scale: scale factor(s) for regularizer.
    Returns:
        regularizer for tf.
    """
    if reg_type == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(reg_scale[0])
    elif reg_type == 'l1':
        regularizer = tf.contrib.layers.l1_regularizer(reg_scale[0])
    elif reg_type == 'no_reg':
        regularizer = lambda x : None
    else:
        raise ValueError("Invalid reg_type[%s]" % reg_type)
    return regularizer


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


## from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

## from elmo https://github.com/allenai/bilm-tf
def average_gradients_v2(tower_grads):
    """
    For sparse gradients.
    """
    from tensorflow.python.training.optimizer import _deduplicate_indexed_slices

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

        g0, v0 = grad_and_vars[0]
        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue
        elif isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)
        else:
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def clip_grad(grads_and_vars, clip):
    """
    Clip gradients by global norm.

    Args:
        grads_and_vars: list of (gradient, variable) tuples.
        clip: global norm.

    Returns:
        Clipped grad_and_vars.
    """
    grad_list = [g for g, v in grads_and_vars]
    var_list  = [v for g, v in grads_and_vars]
    clipped_grads, norm = tf.clip_by_global_norm(grad_list, clip)
    return list(zip(clipped_grads, var_list))

