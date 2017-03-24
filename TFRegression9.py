# -*- coding: utf8 -*-
"""EEG Regression - Core TensorFlow implementation code.

July 2016
March 2017 - update Enea, integrated the queue.

"""

import gc
import math
import re
import resource
import sys
import time

import matplotlib.pyplot as plot
import numpy as np
import numpy.matlib
import scipy.io as sio
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn


def LagData(data, lags):
    """Add temporal context to an array of observations. The observation data has
    a size of N observations by D feature dimensions. This routine returns the new
    data with context, and also array of good rows from the original data. This list
    is important because depending on the desired temporal lags, some of the
    original rows are dropped because there is not enough data.
    
    Using negative lags grab data from earlier (higher) in the data array. While
    positive lags are later (lower) in the data array.
    """
    num_samples = data.shape[0]  # Number of samples in input data
    orig_features_count = data.shape[1]
    new_features_count = orig_features_count * len(lags)
    
    # We reshape the data into a array of size N*D x 1.  And we enhance the lags
    # to include all the feature dimensions which are stretched out in "time".
    unwrapped_data = data.reshape((-1, 1))
    expanded_lags = (lags * orig_features_count).reshape(1, -1).astype(int)
    
    # Now expand the temporal lags array to include all the new feature dimensions
    offsets = numpy.matlib.repmat(expanded_lags, orig_features_count, 1) + \
              numpy.matlib.repmat(np.arange(orig_features_count).reshape(orig_features_count,
                                                                         1), 1,
                                  expanded_lags.shape[0])
    offsets = offsets.T.reshape(1, -1)
    
    indices = numpy.matlib.repmat(offsets, num_samples, 1)
    hops = np.arange(0, num_samples).reshape(-1, 1) * orig_features_count
    hops = numpy.matlib.repmat(hops, 1, hops.shape[1])
    if 0:
        print "Offsets for unwrapped features:", offsets
        print "Offset indices:", indices
        print "Starting hops:", hops
        print "Indices to extract from original:", indices + hops
    
    new_indices = offsets + hops
    good_rows = numpy.where(numpy.all((new_indices >= 0) &
                                      (new_indices < unwrapped_data.shape[0]),
                                      axis=1))[0]
    new_indices = new_indices[good_rows, :]
    new_data = unwrapped_data[new_indices].reshape((-1, new_features_count))
    return new_data, good_rows


def TestLagData():
    """Use some simple data to make sure that the LagData routine is working."""
    input_data = np.arange(20).reshape((10, 2))
    print "Input array:", input_data
    (new_data, good_rows) = LagData(input_data, np.arange(-1, 2))
    print "Output array:", new_data
    print "Good rows:", good_rows


# Use TF to compute the Pearson Correlation of a pair of 1-dimensional vectors.
# From: https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
def PearsonCorrelationTF(x, y, prefix='pearson'):
    """Create a TF network that calculates the Pearson Correlation on two input
    vectors.  Returns a scalar tensor with the correlation [-1:1]."""
    with tf.name_scope(prefix):
        n = tf.to_float(tf.shape(x)[0])
        x_sum = tf.reduce_sum(x)
        y_sum = tf.reduce_sum(y)
        xy_sum = tf.reduce_sum(tf.multiply(x, y))
        x2_sum = tf.reduce_sum(tf.multiply(x, x))
        y2_sum = tf.reduce_sum(tf.multiply(y, y))
        
        r_num = tf.subtract(tf.multiply(n, xy_sum), tf.multiply(x_sum, y_sum))
        r_den_x = tf.sqrt(tf.subtract(tf.multiply(n, x2_sum), tf.multiply(x_sum, x_sum)))
        r_den_y = tf.sqrt(tf.subtract(tf.multiply(n, y2_sum), tf.multiply(y_sum, y_sum)))
        r = tf.div(r_num, tf.multiply(r_den_x, r_den_y), name='r')
    return r


def ComputePearsonCorrelation(x, y):
    """Compute the Pearson's correlation between two numpy vectors (1D only)"""
    n = x.shape[0]
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    xy_sum = np.sum(x * y)
    x2_sum = np.sum(x * x)
    y2_sum = np.sum(y * y)
    r_num = n * xy_sum - x_sum * y_sum
    r_den_x = math.sqrt(n * x2_sum - x_sum * x_sum)
    r_den_y = math.sqrt(n * y2_sum - y_sum * y_sum)
    
    return r_num / (r_den_x * r_den_y)


# Code to check the Pearson Correlation calculation.  Create random data, 
# calculate its correlation, and output the data in a form that is easy to
# paste into Matlab.  Also compute the correlation with numpy so we can compare. 
# Values should be identical.
def TestPearsonCorrelation(n=15):
    x = tf.to_float(tf.random_uniform([n], -10, 10, tf.int32))
    y = tf.to_float(tf.random_uniform([n], -10, 10, tf.int32))
    init = tf.initialize_all_variables()
    
    r = PearsonCorrelationTF(x, y)
    
    borg_session = ''  # 'localhost:' + str(FLAGS.brain_port)
    with tf.Session(borg_session) as sess:
        sess.run(init)
        x_data, y_data, r_data = sess.run([x, y, r], feed_dict={})
        print 'x=', x_data, ';'
        print 'y=', y_data, ';'
        print 'r=', r_data, ';'
        print 'Expected r is', ComputePearsonCorrelation(x_data, y_data)


# noinspection PyAttributeOutsideInit
class RegressionNetwork:
    """Basic class implementing TF Regression."""
    
    def __init__(self):
        self.Clear()
    
    def Clear(self):
        self.g = None
        self.num_hidden = 0
        self.training_steps = 0
        self.x1 = None
        self.W1 = None
        self.b1 = None
        self.y1 = None
        self.W2 = None
        self.b2 = None
        self.y2 = None
        self.ytrue = None
        self.loss = None
        self.optimizer = None
        self.train = None
        self.init = None
        self.save = None
        self.merge_summaries = None
        self.session = None
        self.queue = None
        self.enqueue_op = None
        self.x_input = None
        self.y_input = None
    
    def CreatePrefetchGraph(self):
        """Create the pieces of the graph we need to fetch the data. The two
        primary outputs of this routine are stored in the class variables:
        self.x1 (the input data) and self.ytrue (the predictions)"""
        # From the first answer at:
        #  http://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow
        
        # Input features are length input_d vectors of floats.
        # Output data are length output_d vectors of floats.
        self.x_input, self.y_input = tf.py_func(self.FindRandomData,
                                                [], [tf.float32, tf.float32])
        self.queue = tf.FIFOQueue(10, [tf.float32, tf.float32],
                                  shapes=[[self.batch_size, self.n_steps, self.input_d], [self.batch_size, 1]])
        self.enqueue_op = self.queue.enqueue([self.x_input, self.y_input])
        # Insert data here if feeding the network with a feed_dict.
        self.x1, self.ytrue = self.queue.dequeue()
    
    def CreateEvalGraph(self, output_d, correlation_loss, learning_rate=0.01):
        """Given the true and predicted y values, add all the evaluation and
        summary pieces to the end of the graph."""
        if correlation_loss:
            if output_d > 1:
                raise ValueError("Can't do correlation on multidimensional output")
            # Compute the correlation
            r = PearsonCorrelationTF(self.ytrue, self.ypredict)
            self.loss = tf.negative(r, name='loss_pearson')
        else:
            # Minimize the mean squared errors.
            self.loss = tf.reduce_mean(tf.square(self.ypredict - self.ytrue),
                                       name='loss_euclidean')
        tf.summary.scalar('loss', self.loss)
        #
        # https://www.quora.com/Which-optimizer-in-TensorFlow-is-best-suited-for-learning-regression
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.merged_summaries = tf.summary.merge_all()
    
    # Create a TF network to find values with a single level network that predicts
    # y_data from x_data.  Only set the correlation_loss argument to true if
    # predicting one-dimensional data.
    def CreateEntireNetwork(self, input_d, output_d, n_steps, num_hidden=20,
                            learning_rate=0.01, correlation_loss=False,
                            batch_size=128):
        self.input_d = input_d
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.g = tf.Graph()
        print "Creating a RegressionNetwork with %d hidden units." % num_hidden
        with self.g.as_default():
            self.CreatePrefetchGraph()
            self.ypredict = self.CreateComputation(self.x1, input_d, output_d)
            self.CreateEvalGraph(output_d, correlation_loss)
    
    def CreateComputation(self, x1, input_d, output_d, num_hidden=20):
        # W1 needs to be input_d x num_hidden
        self.W1 = tf.Variable(tf.random_uniform([input_d, num_hidden], -1.0, 1.0),
                              name='W1')
        self.b1 = tf.Variable(tf.zeros([num_hidden]), name='bias1')
        # y1 will be batch_size x num_hidden
        self.y1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x1, self.W1), self.b1),
                             name='y1')
        
        self.W2 = tf.Variable(tf.random_uniform([num_hidden, output_d], -1.0, 1.0),
                              name='W2')
        self.b2 = tf.Variable(tf.zeros([output_d]), name='b2')
        # Output y2 Will be batch_size x output_d
        self.y2 = tf.nn.bias_add(tf.matmul(self.y1, self.W2), self.b2, name='y2')
        return self.y2
    
    def CreateSession(self, session_target=''):
        if not self.session:
            self.session = tf.Session(session_target, graph=self.g)
    
    def FindRandomData(self):
        """Find some data for training. Make sure that these three class varibles
        are set up before this runs: training_batch_size, training_x_data, and
        training_y_data. This code is called at run time, directly by tensorflow
        to get some new data for training. This will be used in a py_func."""
        ri = numpy.floor(numpy.random.rand(
            self.training_batch_size) *
                         self.training_x_data.shape[0]).astype(int)
        training_x = self.training_x_data[ri, :]
        training_y = self.training_y_data[ri, :]
        return training_x.astype(np.float32), training_y.astype(np.float32)
    
    def TrainFromQueue(self, x_data, y_data, batch_size=40, training_steps=6000,
                       reporting_fraction=0.1, session_target='',
                       tensorboard_dir='/tmp/tf'):
        """Train a DNN Regressor, using the x_data to predict the y_data."""
        self.training_x_data = x_data
        self.training_y_data = y_data
        self.training_batch_size = batch_size
        qr = tf.train.QueueRunner(self.queue, [self.enqueue_op] * 4)
        self.CreateSession(session_target)
        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(self.session, coord=coord, start=True)
        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', self.g)
        self.session.run(self.init)
        
        total_time = 0.0
        loss_value = 0.0
        for step in xrange(training_steps):
            if coord.should_stop():
                break
            tic = time.time()
            _, loss_value, summary_values = self.session.run([self.train, self.loss,
                                                              self.merged_summaries])
            total_time += time.time() - tic
            if step % int(training_steps * reporting_fraction) == 0:
                print step, loss_value
            train_writer.add_summary(summary_values, step)
        self.training_steps = training_steps
        print "TrainFromQueue: %d steps averaged %gms per step." % \
              (training_steps, total_time / training_steps * 1000)
        coord.request_stop()
        coord.join(enqueue_threads)
        return 0, loss_value
    
    def TrainWithFeed(self, x_data, y_data, batch_size=40, training_steps=6000,
                      reporting_fraction=0.1, session_target='',
                      tensorboard_dir='/tmp/tf'):
        """Train a DNN Regressor, using the x_data to predict the y_data.
        Report periodically, every
          training_steps*reporting_fraction
        epochs.  The loss function is Euclidean (L2) unless the
        correlation_loss parameter is true.  Output the final model
        to the model_save_file.  """
        self.CreateSession(session_target)
        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', self.g)
        self.session.run(self.init)
        total_time = 0.0
        loss_value = 0.0
        y2_value = 0.0
        for step in xrange(training_steps):
            # Choose some data at random to feed the graph.
            ri = numpy.floor(numpy.random.rand(batch_size) * x_data.shape[0]).astype(int)
            training_x = x_data[ri, :]
            training_y = y_data[ri, :]
            tic = time.time()
            _, loss_value, y2_value, summary_values = \
                self.session.run([self.train, self.loss, self.y2, self.merged_summaries],
                                 feed_dict={self.x1: training_x, self.ytrue: training_y})
            total_time += time.time() - tic
            if step % int(training_steps * reporting_fraction) == 0:
                print step, loss_value  # , training_x.shape, training_y.shape
                print step, loss_value  # , training_x.shape, training_y.shape
            train_writer.add_summary(summary_values, step)
        self.training_steps = training_steps
        print "Average time per training session is %gms." % \
              (1000 * total_time / training_steps)
        return y2_value, loss_value
    
    def SaveState(self, model_save_file, session_target=''):
        self.CreateSession(session_target)
        self.saver.save(self.session, model_save_file)
    
    def RestoreState(self, model_save_file=None, session_target=''):
        self.CreateSession(session_target)
        if model_save_file:
            print "RegressionNetwork.Eval: Restoring the model from:", model_save_file
            self.saver.restore(self.session, model_save_file)
    
    def Eval(self, x_data, y_truth=None, session_target='', tensorboard_dir='/tmp/tf'):
        self.CreateSession(session_target)
        if y_truth is None:
            [y2_value] = self.session.run([self.y2], feed_dict={self.x1: x_data})
        else:
            print "Evaluating the eval loss to", tensorboard_dir
            eval_writer = tf.summary.FileWriter(tensorboard_dir + '/eval', self.g)
            [y2_value, summary_values, loss] = \
                self.session.run([self.y2, self.merged_summaries, self.loss],
                                 feed_dict={self.x1: x_data, self.ytrue: y_truth})
            print "loss is:", loss, "y2_value is:", y2_value, "summary_values are:", summary_values
            eval_writer.add_summary(summary_values, self.training_steps)
        return y2_value
    
    def __enter__(self):
        return self
    
    def Close(self):
        print "RegressorNetwork::Close called."
        if self.session:
            print "   Closing the session too."
            self.session.close()
            self.session = None
        tf.reset_default_graph()
        self.g = None
        self.training_x_data = None
        self.training_y_data = None
    
    def __del__(self):
        """Called when the GC finds no references to this object."""
        print "RegressorNetwork::__del__ called."
        self.Close()
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Called when the variable goes out of scope (like in a with)"""
        print "RegressorNetwork::__exit__ called."
        self.Close()


# noinspection PyAttributeOutsideInit
class DNNRegressionNetwork(RegressionNetwork):
    def CreateEntireNetwork(self, input_d, output_d, num_hidden=6, learning_rate=0.0,
                            rnn_unit='gru', out_layers='[]',
                            activation='sigmoid', bi_rnn=False,
                            opt='adam', init='glorot', n_layers=1,
                            n_steps=32, correlation_loss=False, batch_size=128):
        self.num_hidden = num_hidden
        self.input_d = input_d
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.g = tf.Graph()
        print "Creating a RegressionNetwork with %d hidden units." % num_hidden
        with self.g.as_default():
            self.CreatePrefetchGraph()
            # Inputs
            # x = tf.placeholder(tf.float32, [None, dataset.n_steps, dataset.n_features_in])
            
            # Keeping probability for dropout
            keep_prob = tf.placeholder(tf.float32)
            
            # Weights for the output layers
            levels = [int(a) for a in re.findall(r"[\w']+", out_layers)]
            levels.append(output_d)
            weights, biases, weights_to_merge = {}, {}, []
            
            for k in xrange(len(levels)):
                if k is 0:
                    if bi_rnn:
                        if init == 'uni':
                            weights['hidden{}'.format(k)] = \
                                tf.Variable(tf.random_uniform([num_hidden * 2,
                                                               levels[k]], -.1, .1))
                        elif init == 'gauss':
                            weights['hidden{}'.format(k)] = \
                                tf.Variable(tf.random_normal([num_hidden * 2,
                                                              levels[k]], stddev=.1))
                        elif init == 'glorot':
                            weights['hidden{}'.format(k)] = \
                                tf.get_variable('hidden{}'.format(k),
                                                shape=[num_hidden * 2, levels[k]],
                                                initializer=tf.contrib.layers.xavier_initializer())
                    else:
                        if init == 'uni':
                            weights['hidden{}'.format(k)] = \
                                tf.Variable(tf.random_uniform([num_hidden * n_steps,
                                                               levels[k]], -.1, .1))
                        elif init == 'gauss':
                            weights['hidden{}'.format(k)] = \
                                tf.Variable(tf.random_normal([num_hidden * n_steps,
                                                              levels[k]], stddev=.1))
                        elif init == 'glorot':
                            weights['hidden{}'.format(k)] = \
                                tf.get_variable('hidden{}'.format(k),
                                                shape=[num_hidden * n_steps, levels[k]],
                                                initializer=tf.contrib.layers.xavier_initializer())
                    biases['hidden{}'.format(k)] = tf.Variable(tf.zeros([levels[k]]))
                else:
                    if init == 'uni':
                        weights['hidden{}'.format(k)] = \
                            tf.Variable(tf.random_uniform([levels[k - 1], levels[k]], -.1, .1))
                    elif init == 'gauss':
                        weights['hidden{}'.format(k)] = \
                            tf.Variable(tf.random_normal([levels[k - 1], levels[k]], stddev=.1))
                    elif init == 'glorot':
                        weights['hidden{}'.format(k)] = \
                            tf.get_variable('hidden{}'.format(k),
                                            shape=[levels[k - 1], levels[k]],
                                            initializer=tf.contrib.layers.xavier_initializer())
                    biases['hidden{}'.format(k)] = tf.Variable(tf.zeros([levels[k]]))
                weights_to_merge.append(tf.summary.histogram("weight_hidden{}".format(k),
                                                             weights['hidden{}'.format(k)]))
                weights_to_merge.append(tf.summary.histogram("bias_hidden{}".format(k),
                                                             biases['hidden{}'.format(k)]))
            
            # Register weights to be monitored by tensorboard
            # Let's define the training and testing operations
            self.ypredict, summ_outputs = self.rnn_step(self.x1, weights, biases,
                                                        _keep_prob=keep_prob,
                                                        num_hidden=num_hidden,
                                                        rnn_unit=rnn_unit,
                                                        n_layers=n_layers,
                                                        activation=activation)
            
            weights_to_merge += summ_outputs
        
        # self.ypredict = self.CreateComputation(self.x1, input_d, output_d)
        self.CreateEvalGraph(output_d, correlation_loss)
    
    def rnn_step(self, _input, _weights, _biases, _keep_prob, num_hidden,
                 rnn_unit, n_layers, activation):
        """
        :param _input: a 'Tensor' of shape [batch_size x n_steps x n_features] representing
                   the input to the network
        :param _weights: Dictionary of weights to calculate the activation of the
                fully connected layer(s) after the RNN
        :param _biases: Dictionary of weights to calculate the activation of the
                fully connected layer(s) after the RNN
        :param _keep_prob: float in [0, 1], keeping probability for Dropout
        :param num_hidden: int, number of units in each layer
        :param rnn_unit: string, type of unit can be 'gru', 'lstm' or 'simple'
        :param n_layers: int, number of layers in the RNN
        :param activation: string, activation for the fully connected layers:
                   'linear', 'relu, 'sigmoid', 'tanh'
        :return: output of the network
        """
        
        hist_outputs = []
        prev_rec = _input
        if rnn_unit == 'lstm':
            cell = MultiRNNCell([LSTMCell(num_hidden, use_peepholes=True)] * n_layers)
        
        elif rnn_unit == 'gru':
            cell = MultiRNNCell([GRUCell(num_hidden)] * n_layers)
        
        else:
            cell = MultiRNNCell([BasicRNNCell(num_hidden)] * n_layers)
        
        prev_act, prev_hid = dynamic_rnn(cell, inputs=prev_rec, dtype=tf.float32)
        prev_act = tf.reshape(prev_act, (-1, self.num_hidden * self.n_steps))
        
        for k in xrange(len(_weights) - 1):
            hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(prev_act,
                                                         _weights['hidden{}'.format(k)]),
                                               _biases['hidden{}'.format(k)]))
            
            hist_outputs.append(tf.summary.histogram("FC_{}".format(k), hidden))
            
            hid_drop = tf.nn.dropout(hidden, _keep_prob)
            prev_act = hid_drop
        
        last_act = tf.nn.bias_add(tf.matmul(prev_act,
                                            _weights['hidden{}'.format(len(_weights) - 1)]),
                                  _biases['hidden{}'.format(len(_weights) - 1)])
        
        hist_outputs.append(tf.summary.histogram("last_act", last_act))
        
        if activation == "linear":
            ret_act = last_act
        elif activation == 'relu':
            ret_act = tf.nn.relu(last_act)
        elif activation == 'sigmoid':
            ret_act = tf.nn.sigmoid(last_act)
        elif activation == 'tanh':
            ret_act = tf.nn.tanh(last_act)
        else:
            raise ValueError("Activation requested not yet implemented, choose between "
                             "'linear', "
                             "'relu', "
                             "'sigmoid' or "
                             "'tanh'")
        
        return ret_act, hist_outputs


"""Polynomial Fitting.
Now generate some fake test data and make sure we can properly regress the data.
This is just a polynomial fitting.
"""


def TestPolynomialFitting():
    # Create 8000 phony x and y data points with NumPy
    x_data = 1.2 * np.random.rand(8000, 1).astype("float32") - 0.6
    
    # Create point-wise 3rd order non-linearity
    y_data = (x_data - .5) * x_data * (x_data + 0.5) + 0.3
    
    # First, regress with Euclidean loss function
    with DNNRegressionNetwork() as regressor:
        regressor.CreateEntireNetwork(x_data.shape[1], y_data.shape[1],
                                      learning_rate=0.01,
                                      correlation_loss=False,
                                      batch_size=128)
        _, _ = regressor.TrainWithFeed(x_data, y_data)  # could return loss value
        y_prediction = regressor.Eval(x_data)
    
    plot.clf()
    plot.plot(x_data, y_prediction, '.', x_data, y_data, '.')
    plot.xlabel('Input Variable')
    plot.ylabel('Output Variable')
    plot.legend(('Prediction', 'Truth'))
    plot.title('DNN Regression with Euclidean Loss')
    plot.show()
    
    # Second, regress with a correlation loss function
    with RegressionNetwork() as regressor:
        regressor.CreateEntireNetwork(x_data.shape[1], y_data.shape[1],
                                      learning_rate=0.01,
                                      correlation_loss=True,
                                      n_steps=1)
        _, _ = regressor.TrainWithFeed(x_data, y_data)  # could return loss value
        y_prediction = regressor.Eval(x_data)
    
    plot.clf()
    plot.plot(x_data, y_prediction, '.', x_data, y_data, '.')
    plot.xlabel('Input Variable')
    plot.ylabel('Output Variable')
    plot.legend(('Prediction', 'Truth'))
    plot.title('DNN Regression with Correlation Loss')
    plot.show()


def LoadTellurideDemoData(demo_data_loc):
    demodata = sio.loadmat(demo_data_loc)
    print "Data keys in Matlab input file:", sorted(demodata.keys())
    fsample = demodata['fsample'][0][0]
    eeg_data = (demodata['eeg'].reshape((32)))
    for i in xrange(eeg_data.shape[0]):
        eeg_data[i] = eeg_data[i].astype(np.float32)
    audio_data = demodata['wav'].reshape((4))
    for i in xrange(audio_data.shape[0]):
        audio_data[i] = audio_data[i].astype(np.float32)
    print "Audio data has size:", audio_data.shape, audio_data[0].shape
    print "EEG data has size:", eeg_data.shape, eeg_data[0].shape
    return audio_data, eeg_data, fsample


def AssembleDemoData(audio_data, eeg_data, trials, max_lag):
    lags = np.arange(0, int(max_lag))
    all_eeg = None
    all_audio = None
    for t in trials:
        # don't wanna lag all the data
        (laggedEEG, good_eeg_rows) = LagData(eeg_data[t], lags)
        laggedAudio = audio_data[t % 4][good_eeg_rows, 0].reshape((-1, 1))
        if all_eeg is None:
            all_eeg = laggedEEG
        else:
            all_eeg = np.append(all_eeg, laggedEEG, 0)
        if all_audio is None:
            all_audio = laggedAudio
        else:
            all_audio = np.append(all_audio, laggedAudio, 0)
    print "AssembleDemoData:", all_audio.shape, all_eeg.shape
    return all_audio, all_eeg


def TestNumberHidden(audio_data, eeg_data, fsample, hidden_list=None,
                     session_target='', training_steps=5000, batch_size=1000,
                     tensorboard_base='/tmp/tf', n_steps=32):
    if hidden_list is None:
        hidden_list = [6]
    num_trials = eeg_data.shape[0]
    frac_correct = np.zeros((num_trials, max(hidden_list) + 1))
    train_loss = np.zeros((num_trials, max(hidden_list) + 1))
    
    for hidden in hidden_list:
        for t in range(num_trials):
            test_set = [t]
            train_set = list(set(range(num_trials)).difference(test_set))
            # lag after not now
            all_audio, all_eeg = AssembleDemoData(audio_data, eeg_data,
                                                  np.array(train_set), fsample * 0.25)
            test_audio, test_eeg = AssembleDemoData(audio_data, eeg_data,
                                                    np.array(test_set), fsample * 0.25)
            print "Before DNNRegression:", all_eeg.shape, all_audio.shape, \
                test_eeg.shape, test_audio.shape
            with RegressionNetwork() as regressor:
                tensorboard_dir = '%s/telluride-%03d' % (tensorboard_base, t)
                regressor.CreateEntireNetwork(all_eeg.shape[1], all_audio.shape[1], n_steps=n_steps,
                                              learning_rate=1, correlation_loss=True,
                                              num_hidden=hidden, batch_size=batch_size)
                (y_prediction, loss_value) = regressor.TrainFromQueue(all_eeg, all_audio,
                                                                      batch_size=batch_size,
                                                                      training_steps=training_steps,
                                                                      tensorboard_dir=tensorboard_dir,
                                                                      session_target=session_target)
                audio_predict = regressor.Eval(test_eeg, test_audio,
                                               tensorboard_dir=tensorboard_dir)
            gc.collect()
            print "At end of TestNumberHidden: MaxRSS is", \
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            c = np.corrcoef(audio_predict.T, test_audio[0:audio_predict.shape[0], :].T)
            frac_correct[t, hidden] = c[0, 1]
            train_loss[t, hidden] = loss_value
            print frac_correct
            sys.stdout.flush()
    return frac_correct, train_loss


def RunDemoDataTest(num_trials=32, hidden_number=6, batch_size=1000,
                    demo_data_loc='DemoForTF.mat',
                    session_target='', training_steps=5000,
                    tensorboard_base='/tmp/tf', n_steps=32):
    (audio_data, eeg_data, fsample) = LoadTellurideDemoData(demo_data_loc)
    frac_correct, train_loss = TestNumberHidden(audio_data, eeg_data[0:num_trials],
                                                fsample, [hidden_number],
                                                batch_size=batch_size,
                                                session_target=session_target,
                                                training_steps=training_steps,
                                                tensorboard_base=tensorboard_base,
                                                n_steps=n_steps)
    numpy.set_printoptions(linewidth=100000000)
    print frac_correct
    
    # Save the data across all trials into two files so we can plot them later.
    frac_name = 'frac_correct_%02d.txt' % hidden_number
    np.savetxt(frac_name, frac_correct)
    
    loss_name = 'train_loss_%02d.txt' % hidden_number
    np.savetxt(loss_name, train_loss)
    return frac_name, loss_name


def RunSaveModelTest(demo_data_loc='testing/DemoDataForTensorFlow.mat',
                     training_steps=1000, num_trials=3):
    (audio_data, eeg_data, fsample) = LoadTellurideDemoData(demo_data_loc)
    max_lags = fsample * 0.25
    all_audio, all_eeg = AssembleDemoData(audio_data, eeg_data,
                                          np.array([0]), max_lags)
    eeg_data = eeg_data[0:num_trials]  # We don't need all the data to test this.
    model_file = '/tmp/regression_model.tf'
    num_hidden = 6
    with RegressionNetwork() as regressor:
        regressor.CreateEntireNetwork(all_eeg.shape[1], all_audio.shape[1],
                                      learning_rate=1, correlation_loss=True,
                                      num_hidden=num_hidden,
                                      n_steps=1)
        _, _ = regressor.TrainWithFeed(all_eeg, all_audio,
                                                             batch_size=1000,
                                                             training_steps=training_steps)  # could return loss value
        regressor.SaveState(model_file)
    
    with RegressionNetwork() as regressor:
        regressor.CreateEntireNetwork(all_eeg.shape[1], all_audio.shape[1],
                                      learning_rate=1, correlation_loss=True,
                                      num_hidden=num_hidden,
                                      n_steps=1)
        regressor.RestoreState(model_file)
        audio_predict2 = regressor.Eval(all_eeg)
    
    print "Variance of predictions is:", np.var(audio_predict2)
    print


def regression_main(argv):
    print "Testing the PearsonCorrelation TF Graph (#6)"
    if 0:
        print "Testing basic line fitting."
        TestPearsonCorrelation()
    
    if 1:
        print "\nTesting the Polynomial fitting TF Graph"
        TestPolynomialFitting()
    
    if 0:
        print "\nRunning the TFRegression.regression_main() function."
        RunDemoDataTest(hidden_number=3, num_trials=3,
                        batch_size=1000, training_steps=1000, n_steps=32)
    
    if 0:
        print "\nRunning the TFRegression.regression_main() function."
        RunSaveModelTest(training_steps=1000)


if __name__ == "__main__":
    regression_main(sys.argv)
