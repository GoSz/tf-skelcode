#!/usr/bin/python
# -*- coding:utf-8 -*-


import sys
import time
import tensorflow as tf
import numpy as np

from utils.common import *
from text_model.data import *
from text_model.utils import *


class TextModelDataset(object):
    """
    A wrapper class for dataset.
    """
    def __init__(self, options, is_training, gpu_num):
        self.is_training = is_training
        self.gpu_num = gpu_num
        self.need_evaluate = is_training and options.get("need_evaluate", False)
        self.batch_size  = options["batch_size"]

        need_reserve = True if options.get('pre_trained_wembed', None) == None else False
        self.vocab = Vocabulary(options["vocab_file"], need_reserve=need_reserve)

        need_shuffle = True if self.is_training else False
        train_set = self._init_tfrec_dataset(options["train_file"], need_shuffle)
        self.iterator = tf.data.Iterator.from_structure(train_set.output_types,
                                                        train_set.output_shapes)
        ## ops for (re)init train dataset
        self.data_init_op = self.iterator.make_initializer(train_set)

        if self.is_training and self.need_evaluate:
            val_set = self._init_tfrec_dataset(options["val_file"], False)
            ## ops for (re)init validation dataset
            self.data_init_op_val = self.iterator.make_initializer(val_set)

    def _init_tfrec_dataset(self, data_file, need_shuffle):
        """
        Get dataset from tf record file.
        """
        dataset = tf.data.TFRecordDataset(data_file)
        def parse_func(example_proto):
            ## NOTE define proto parse function for tfrecord here
            proto_dict = {}
            parsed_features = tf.parse_single_example(example_proto, proto_dict)
            return parsed_features
        dataset = dataset.prefetch(buffer_size=self.batch_size * self.gpu_num)
        if need_shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_size * self.gpu_num * 2)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_func,
                                                              batch_size=self.batch_size,
                                                              num_parallel_batches=self.gpu_num,
                                                              drop_remainder=(self.gpu_num > 1)))
        return dataset


class TextBasedModel(object):
    """
    A skeleton class for text based model.
    """
    def __init__(self, options, dataset, is_training):
        """
        Args:
            options: dict of options.
            dataset: wrapper of tf-dataset,
                if is not `None`, the whole graph can be built directly.
        """
        self.options = options
        self.dataset = dataset
        self.is_training = is_training
        self._load_options()
        self._build()

    def _load_options(self):
        """
        Unpack options.
        """
        ## NOTE load options here
        self.wembed_dim  = self.options['wembed_dim']
        self.vocab_size  = self.options['vocab_size']
        self.dropout     = 0.

    def _build(self):
        """
        Build the model.
        """
        ## NOTE define special placeholders(not for model input)
        self._init_placeholders()

        self._init_word_embedding()

        ## NOTE if dataset is passed in, the whole graph can be built directly
        ## else the model graph should be built manually, by calling `inference`
        if self.dataset != None:
            self._build_tf_graph()

    def _init_placeholders(self):
        """
        Define all placeholders.
        """
        ## make keep_prob a placeholder for convenience
        keep_prob = 1.0 - self.dropout
        if self.is_training:
            self.keep_prob = tf.placeholder_with_default(keep_prob, shape=[])
        else:
            self.keep_prob = tf.placeholder_with_default(1.0, shape=[])

    def _init_word_embedding(self):
        """
        Init word embeddings for text token ids.
        """
        ## use pre-trained word embedding or not
        self.pre_trained_wembed = self.options.get('pre_trained_wembed', None)
        shape = [self.vocab_size, self.wembed_dim]
        with tf.variable_scope("word_embeddings"), tf.device('/cpu:0'):
            if self.pre_trained_wembed:
                self.wembed_init = tf.placeholder(dtype=TF_DTYPE, shape=shape)
                shape = None
                ## finetune pre-trained wembed or not
                trainable = self.options.get('finetune_wembed', False)
            else:
                self.wembed_init = tf.random_normal_initializer()
                trainable = True

            self.word_embed = tf.get_variable("embeddings", shape=shape,
                                              dtype=TF_DTYPE, initializer=self.wembed_init,
                                              trainable=trainable)

    def _build_tf_graph(self):
        """
        Build the whole task defined model graph.
        """
        ## NOTE get model input from dataset
        self._init_input()

        ## NOTE build model graph
        self.model_output = self.inference(self.model_input)

        ## NOTE build loss function
        if self.is_training:
            self.loss_out = self.get_loss(self.model_output, self.model_label)

    def inference(self, *args, **kwargs):
        """
        Build model graph, run model inference with inputs.
        """
        self.model_output = self._inference(*args, **kwargs)
        return self.model_output

    def get_loss(self, *args, **kwargs):
        """
        Get model loss with inference result and labels.
        """
        self.loss_out = self._get_loss(*args, **kwargs)
        return self.loss_out

    def _init_input(self):
        """
        Get model input from dataset.
        """
        ## NOTE manage model inputs with dataset
        self.model_input, self.model_label = self.dataset.iterator.get_next()

    def _inference(self, model_input):
        """
        Run model inference with inputs.
        """
        ## NOTE put model inference logic here
        return { 'pred' : None }

    def _get_loss(self, model_out, label):
        """
        Get model loss with inference results and labels.
        """
        ## NOTE put model loss logic here
        return { 'loss' : None }


def train(options, save_dir, gpu_num=1):
    """
    Train model.

    Args:
        options: dict of options.
        save_dir: directory for saving option file and tf model.

    Returns:
        None
    """
    with tf.device('/cpu:0'):
        ## prepare dataset
        dataset = TextModelDataset(options, is_training=True, gpu_num=gpu_num)
        options['vocab_size'] = dataset.vocab.size

        ## manage learning rate and optimizer
        global_step = tf.get_variable('global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
        decay_steps = options['train']['decay_steps']
        lr = tf.train.exponential_decay(options['train']['start_learning_rate'],
                                        global_step, decay_steps,
                                        options['train']['decay_ratio'],
                                        staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(lr)

        ## build model
        gpu_num = gpu_num if gpu_num >= 1 else 1
        models, tower_grads, losses = [], [], []
        for i in range(gpu_num):
            with tf.variable_scope("model", reuse=(i>0)), tf.device('/gpu:%d' % i):
                model = TextBasedModel(options, dataset=dataset, is_training=True)
                models.append(model)
                tower_grads.append(optimizer.compute_gradients(model.loss_out['loss']))
                losses.append(model.loss_out['loss'])
        grads = tower_grads[0] if gpu_num == 1 else average_gradients_v2(tower_grads)
        loss  = tf.reduce_mean(tf.stack(losses))
        model = models[0]

        ## gradient clip
        grad_clip = options.get('grad_clip', None)
        if grad_clip != None:
            grads = clip_grad(grads, grad_clip)
        opt = optimizer.apply_gradients(grads, global_step=global_step)

        ## all ops for train and validation
        train_ops = { 'opt'  : opt,
                      'loss' : loss }
        val_ops   = { 'loss' : loss }
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

    ## ready to go
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    with tf.Session(config=config) as sess:
        if model.pre_trained_wembed:
            wembed = read_wembed(model.pre_trained_wembed)
            sess.run(init, feed_dict={model.wembed_init: wembed})
            del wembed  ## for gc
        else:
            sess.run(global_init)

        dump_options(options, save_dir+'/options.json')

        max_epoch = options['max_epoch']
        for epoch in range(1, max_epoch+1):
            ## train a epoch
            sess.run(dataset.data_init_op)
            batch_cnt = 0
            t0 = time.time()
            avg_speed = 0.
            while True:
                try:
                    _train_ops = sess.run(train_ops)
                    batch_cnt += (1 * gpu_num)
                    if (batch_cnt % 10) == 0:
                        avg_speed = batch_cnt / (time.time() - t0)
                        sys.stderr.write('\33[2K\r%f batch/sec\tbatch_num=%d\ttrain_loss=%f' % \
                                         (batch_cnt, avg_speed, _train_ops['loss']))
                    assert not np.isnan(_train_ops['loss']), 'model diverged with loss = NaN'
                except tf.errors.OutOfRangeError:
                    sys.stderr.write("\n")
                    break

            ## pass over the validation set per epoch
            if dataset.need_evaluate:
                sys.stderr.write("Epoch[%d/%d]\tevaluating...")
                sess.run(dataset.data_init_op_val)
                val_stats = { 'loss' : [] }
                ## dropout (if exist) should be disabled when evaluating during training
                feed_dict = { model.keep_prob : 1.0 for model in models }
                while True:
                    try:
                        _val_ops = sess.run(val_ops, feed_dict=feed_dict)
                        val_stats['loss'].append(_val_ops['loss'])
                    except tf.errors.OutOfRangeError:
                        break
                val_loss = np.mean(val_stats['loss'])
                sys.stderr.write("\33[2K\rEpoch[%d/%d]\tval_loss=%f\n" % (epoch, max_epoch, val_loss))

            ## do save per epoch
            saver.save(sess, save_dir+"/model", global_step=epoch)

        ## final save
        sys.stderr.write("Train finish\n")
        saver.save(sess, save_dir+"/model")


def predict(option_file, model_path, input_file, output_file):
    """
    Predict with pre-trained tf model.

    Args:
        option_file: json option file for model.
        model_path: tf model file path.
        input_file: predict inputs.
        output_file: predict results.

    Returns:
        None
    """
    options = load_options(option_file)
    ## modify options for predict
    options["train_file"] = input_file
    options["need_evaluate"] = False

    gpu_num = 1
    with tf.device('/cpu:0'):
        ## prepare dataset
        dataset = TextModelDataset(options, is_training=False, gpu_num=gpu_num)
        assert options['vocab_size'] == dataset.vocab.size, "vocabulary size not same"

        ## build model
        models = []
        for i in range(gpu_num):
            with tf.variable_scope("model", reuse=(i>0)), tf.device('/gpu:%d' % i):
                model = TextBasedModel(options, dataset=dataset, is_training=False)
                models.append(model)
        model = models[0]

        ## all ops for predict
        pred_ops = { 'pred'  : model.model_output['pred'] }

    ## ready to go
    saver  = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess, open(output_file, "w") as fout:
        saver.restore(sess, model_path)
        sess.run(dataset.data_init_op)
        batch_cnt = 0
        t0 = time.time()
        avg_speed = 0.
        while True:
            try:
                _pred_ops = sess.run(pred_ops)
                batch_cnt += 1 * gpu_num
                if (batch_cnt % 10) == 0:
                    avg_speed = batch_cnt / (time.time() - t0)
                    sys.stderr.write('\33[2K\r%f batch/sec\tbatch_num=%d' % (batch_cnt, avg_speed))
                ## NOTE output predict results
                fout.write()
            except tf.errors.OutOfRangeError:
                sys.stderr.write("\n")
                break
    sys.stderr.write("Predict finish\n")

