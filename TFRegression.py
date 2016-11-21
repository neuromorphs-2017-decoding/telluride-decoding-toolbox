# -*- coding: utf8 -*-
"""EEG Regression - Core TensorFlow implementation code.

July 2016

"""

import tensorflow as tf
import numpy as np
import math, random, sys, time
import numpy.matlib
import matplotlib.pyplot as plot
import scipy.io as sio   # for loadmat

def LagData(data, lags):
  '''Add temporal context to an array of observations. The observation data has
  a size of N observations by D feature dimensions. This routine returns the new
  data with context, and also array of good rows from the original data. This list
  is important because depending on the desired temporal lags, some of the 
  original rows are dropped because there is not enough data. 
  
  Using negative lags grab data from earlier (higher) in the data array. While
  positive lags are later (lower) in the data array.
  '''
  if type(lags) == list:
    lags = np.array(lags)
  num_samples = data.shape[0]   # Number of samples in input data
  orig_features_count = data.shape[1]
  new_features_count = orig_features_count*len(lags)
  
  # We reshape the data into a array of size N*D x 1.  And we enhance the lags
  # to include all the feature dimensions which are stretched out in "time".
  unwrapped_data = data.reshape((-1, 1))
  expanded_lags = (lags*orig_features_count).reshape(1, -1).astype(int)
  
  # Now expand the temporal lags array to include all the new feature dimensions
  offsets = numpy.matlib.repmat(expanded_lags, orig_features_count, 1) + \
  numpy.matlib.repmat(np.arange(orig_features_count).reshape(orig_features_count, 
                                 1), 1, 
            expanded_lags.shape[0])
  offsets = offsets.T.reshape(1, -1)
  
  indices = numpy.matlib.repmat(offsets, num_samples, 1)
  hops = np.arange(0, num_samples).reshape(-1, 1)*orig_features_count
  hops = numpy.matlib.repmat(hops, 1, hops.shape[1])
  if 0:
    print "Offsets for unwrapped features:", offsets
    print "Offset indices:", indices
    print "Starting hops:", hops
    print "Indices to extract from original:", indices+hops
  
  new_indices = offsets + hops
  good_rows = numpy.where(numpy.all((new_indices >= 0) & 
                  (new_indices < unwrapped_data.shape[0]), 
                  axis=1))[0]
  new_indices = new_indices[good_rows, :]
  new_data = unwrapped_data[new_indices].reshape((-1, new_features_count))
  return new_data, good_rows

def TestLagData():
  input_data = np.arange(20).reshape((10,2))
  print "Input array:", input_data
  (new_data, good_rows) = LagData(input_data, np.arange(-1,2))
  print "Output array:", new_data
  print "Good rows:", good_rows

# Use TF to compute the Pearson Correlation of a pair of 1-dimensional vectors.
# From: https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
def PearsonCorrelationTF(x, y, prefix = 'pearson'):
  '''Create a TF network that calculates the Pearson Correlation on two input
  vectors.  Returns a scalar tensor with the correlation [-1:1].'''
  with tf.name_scope(prefix):
    n = tf.to_float(tf.shape(x)[0])
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)
    xy_sum = tf.reduce_sum(tf.mul(x, y))
    x2_sum = tf.reduce_sum(tf.mul(x, x))
    y2_sum = tf.reduce_sum(tf.mul(y, y))
    
    r_num = tf.sub(tf.mul(n, xy_sum), tf.mul(x_sum, y_sum))
    r_den_x = tf.sqrt(tf.sub(tf.mul(n, x2_sum), tf.mul(x_sum, x_sum)))
    r_den_y = tf.sqrt(tf.sub(tf.mul(n, y2_sum), tf.mul(y_sum, y_sum)))
    r = tf.div(r_num, tf.mul(r_den_x, r_den_y), name='r')
  return r

def ComputePearsonCorrelation(x, y):
  '''Compute the Pearson's correlation between two numpy vectors (1D only)'''
  n = x.shape[0]
  x_sum = np.sum(x)
  y_sum = np.sum(y)
  xy_sum = np.sum(x*y)
  x2_sum = np.sum(x*x)
  y2_sum = np.sum(y*y)
  r_num = n*xy_sum - x_sum*y_sum
  r_den_x = math.sqrt(n*x2_sum - x_sum*x_sum)
  r_den_y = math.sqrt(n*y2_sum - y_sum*y_sum)
  
  return r_num/(r_den_x*r_den_y)
  
# Code to check the Pearson Correlation calculation.  Create random data, 
# calculate its correlation, and output the data in a form that is easy to
# paste into Matlab.  Also compute the correlation with numpy so we can compare. 
# Values should be identical.
def TestPearsonCorrelation(N=15):
  x = tf.to_float(tf.random_uniform([N], -10, 10, tf.int32))
  y = tf.to_float(tf.random_uniform([N], -10, 10, tf.int32))
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

# Create a TF network to find values with a two level network that predicts 
# y_data from x_data.  Only set the correlation_loss argument to true if
# predicting one-dimensional data.
def CreateRegressionNetwork(input_d, output_d, num_hidden=20, 
              learning_rate=0.01, correlation_loss=False):
  g = tf.Graph()
  with g.as_default():
    x1 = tf.placeholder(tf.float32, shape=(None, input_d), name='x1') # Will be batch_size x input_d
    W1 = tf.Variable(tf.random_uniform([input_d,num_hidden], -1.0, 1.0), name='W1')  # input_d x num_hidden
    b1 = tf.Variable(tf.zeros([num_hidden]), name='bias1')
    y1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x1,W1), b1), name='y1') # batch_size x num_hidden
  
    W2 = tf.Variable(tf.random_uniform([num_hidden,output_d], -1.0, 1.0), name='W2')
    b2 = tf.Variable(tf.zeros([output_d]), name='b2')
    y2 = tf.nn.bias_add(tf.matmul(y1,W2), b2, name='y2') # Will be batch_size x output_d
    ytrue = tf.placeholder(tf.float32, shape=(None, output_d), name='ytrue') # num_batch x output_d
  
    if correlation_loss:
      # Compute the correlation
      r = PearsonCorrelationTF(ytrue, y2)
      tf.scalar_summary('correlation', r)
      loss = tf.neg(r, name='loss_pearson')
    else:
      # Minimize the mean squared errors.
      loss = tf.reduce_mean(tf.square(y2 - ytrue), name='loss_euclidean')
      tf.scalar_summary('loss', loss)
  
    # https://www.quora.com/Which-optimizer-in-TensorFlow-is-best-suited-for-learning-regression
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
  
    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    merged_summaries = tf.merge_all_summaries()
  return g, train, loss, init, x1, y2, ytrue, merged_summaries, saver

def TrainDNNRegression(x_data, y_data, x_test=None, y_test=None, num_hidden=20, 
          learning_rate=0.5, training_steps=6000, batch_size=40,
          reporting_fraction=None, correlation_loss=False,
          tensorboard_dir='/tmp', model_save_file=None):
  '''Train a DNN Regressor, using the x_data to predict the y_data.
  If test data is provided, also run the test data and compute its
  correlation so we can monitor for overfitting.  The batch_size
  and training_steps determine how much data is fed in at a time, 
  and for how many epochs. Report periodically, every 
    training_steps*reporting_fraction
  epochs.  The loss function is Euclidean (L2) unless the 
  correlation_loss parameter is true.  Output the final model
  to the model_save_file.  '''
  (g, train, loss, init, x1, y2, ytrue, merged_summaries, saver) = \
    CreateRegressionNetwork(x_data.shape[1],  y_data.shape[1], 
                            num_hidden=num_hidden, 
                            learning_rate=learning_rate, 
                            correlation_loss=correlation_loss)
  borg_session = ''  #'localhost:' + str(FLAGS.brain_port)
  y2_test_value = None
  with tf.Session(borg_session, graph=g) as sess:
    train_writer = tf.train.SummaryWriter(tensorboard_dir + '/train', g)
    test_writer = tf.train.SummaryWriter(tensorboard_dir + '/test')
    sess.run(init)
    if reporting_fraction is None:
      reporting_fraction = .1
    # Train the model for training_steps epochs
    for step in xrange(training_steps):
      ri = numpy.floor(numpy.random.rand(batch_size)*x_data.shape[0]).astype(int)
      training_x = x_data[ri,:]
      training_y = y_data[ri,:]
      # print training_x.shape, training_y.shape
      _, loss_value, y2_value, summary_values = \
        sess.run([train, loss, y2, merged_summaries],
                 feed_dict={x1: training_x, ytrue: training_y})
      train_writer.add_summary(summary_values, step)
      if step % int(training_steps*reporting_fraction) == 0:
        print step, loss_value # , training_x.shape, training_y.shape
        # Run the model over the test data.
        if x_test is not None:
          # print "x_test is", x_test.shape, " y_test is", y_test.shape
          (y2_value, summary_values) = sess.run([y2, merged_summaries], 
                                                feed_dict={x1: x_test, 
                                                           ytrue: y_test})
          test_writer.add_summary(summary_values, step)
      if model_save_file:
        save_path = saver.save(sess, model_save_file)
    if x_test is not None:
      # print "x_test is", x_test.shape, " y_test is", y_test.shape
      (y2_value, summary_values) = sess.run([y2, merged_summaries], 
                                            feed_dict={x1: x_test, 
                                                       ytrue: y_test})
    return y2_value, loss_value

def EvalDNNRegression(x_data, model_save_file, output_dimensions=1, num_hidden=20, 
                      correlation_loss=False):
  (g, train, loss, init, x1, y2, ytrue, merged_summaries, saver) = \
  CreateRegressionNetwork(x_data.shape[1], output_dimensions, 
              num_hidden=num_hidden, learning_rate=0, 
              correlation_loss=correlation_loss)
  borg_session = ''
  with tf.Session(borg_session, graph=g) as sess:
    print "EvalDNNRegression: Restoring the model from:", model_save_file
    saver.restore(sess, model_save_file)
    sess.run(init)
    [y2_value] = sess.run([y2], feed_dict={x1: x_data})
    print "In EvalDNNRegression, y2_value type is", type(y2_value)
    return y2_value

"""Polynomial Fitting.

Now generate some fake test data and make sure we can properly regress the data.
This is just a polynomial fitting.
"""

def TestPolynomialFitting():
  # Create 8000 phony x and y data points with NumPy
  x_data = 1.2*np.random.rand(8000,1).astype("float32")-0.6

  # Create pointwise 3rd order nonlinearity
  y_data = (x_data - .5)*x_data*(x_data+0.5) + 0.3

  (y2out, loss_value) = TrainDNNRegression(x_data, y_data, x_data, learning_rate=0.01)

  plot.clf()
  plot.plot(x_data, y2out, '.', x_data, y_data, '.')
  plot.xlabel('Input Variable')
  plot.ylabel('Output Variable')
  plot.legend(('Prediction', 'Truth'))
  plot.title('DNN Regression with Euclidean Loss')
  plot.show()

  (y2out, loss_value) = TrainDNNRegression(x_data, y_data, x_data, correlation_loss=True)

  plot.clf()
  plot.plot(x_data, y2out, '.', x_data, y_data, '.')
  plot.xlabel('Input Variable')
  plot.ylabel('Output Variable')
  plot.legend(('Prediction', 'Truth'))
  plot.title('DNN Regression with Correlation Loss')
  plot.show()

def LoadTellurideDemoData(demo_data_loc):
  demodata = sio.loadmat(demo_data_loc)
  print sorted(demodata.keys())
  fsample = demodata['fsample'][0][0]
  eeg_data = demodata['eeg'].reshape((32))
  audio_data = demodata['wav'].reshape((4))
  print audio_data.shape, audio_data[0].shape
  print eeg_data.shape, eeg_data[0].shape
  return (audio_data, eeg_data, fsample)
  
def AssembleDemoData(audio_data, eeg_data, trials, max_lag):
  lags = np.arange(0, int(max_lag))
  all_eeg = None
  all_audio = None
  for t in trials:
    (laggedEEG, good_eeg_rows) = LagData(eeg_data[t], lags)
    laggedAudio = audio_data[t%4][good_eeg_rows,0].reshape((-1,1))
    if all_eeg == None:
      all_eeg = laggedEEG
    else:
      all_eeg = np.append(all_eeg, laggedEEG, 0)
    if all_audio == None:
      all_audio = laggedAudio
    else:
      all_audio = np.append(all_audio, laggedAudio, 0)
  print "AssembleDemoData:", all_audio.shape, all_eeg.shape
  return all_audio, all_eeg

def TestNumberHidden(audio_data, eeg_data, fsample, hidden_list=[6]):
  num_trials = eeg_data.shape[0]
  frac_correct = np.zeros((num_trials, max(hidden_list)+1))
  train_loss = np.zeros((num_trials, max(hidden_list)+1))
  
  for hidden in hidden_list:
    for t in range(num_trials):
      test_set = [t]
      train_set = list(set(range(num_trials)).difference(test_set))
      max_lags = fsample*0.25
      all_audio, all_eeg = AssembleDemoData(audio_data, eeg_data, 
                                            np.array(train_set), max_lags)
      test_audio, test_eeg = AssembleDemoData(audio_data, eeg_data, 
                                            np.array(test_set), max_lags)
      print "Before TrainDNNRegression:", \
            all_eeg.shape, all_audio.shape, \
            test_eeg.shape, test_audio.shape
      audioPredict, loss = TrainDNNRegression(all_eeg, all_audio, test_eeg, test_audio, 
                   learning_rate=1, num_hidden=hidden,
                   reporting_fraction=.1, training_steps=1000,
                   batch_size=1000, correlation_loss=True,
                   tensorboard_dir='/tmp/telluride-%03d' % t)
      print "Before corrcoef:", audioPredict.shape, test_audio.shape
      c = np.corrcoef(audioPredict.T, test_audio[0:audioPredict.shape[0],:].T)
      frac_correct[t, hidden] = c[0,1]
      train_loss[t, hidden] = loss
      print frac_correct
      sys.stdout.flush()
  return frac_correct, train_loss

def RunDemoDataTest(num_trials = 32, hidden_number = 6, 
                    demo_data_loc = 'testing/DemoDataForTensorFlow.mat'):
  (audio_data, eeg_data, fsample) = LoadTellurideDemoData(demo_data_loc)
  frac_correct, train_loss = TestNumberHidden(audio_data, eeg_data[0:num_trials], 
                                              fsample, [hidden_number])
  numpy.set_printoptions(linewidth=100000000)
  print frac_correct
  
  # Save the data across all trials into two files so we can plot them later.
  frac_name = 'frac_correct_%02d.txt' % hidden_number
  np.savetxt(frac_name, frac_correct)

  loss_name = 'train_loss_%02d.txt' % hidden_number
  np.savetxt(loss_name, train_loss)
  return frac_name, loss_name

  
def RunSaveModelTest(demo_data_loc = 'testing/DemoDataForTensorFlow.mat'):
  (audio_data, eeg_data, fsample) = LoadTellurideDemoData(demo_data_loc)
  max_lags = fsample*0.25
  all_audio, all_eeg = AssembleDemoData(audio_data, eeg_data, 
                                        np.array([0]), max_lags)
  hidden = 6
  model_file = '/tmp/regression_model.tf'
  _, loss = TrainDNNRegression(all_eeg, all_audio, None, None, 
               learning_rate=1, num_hidden=hidden,
               reporting_fraction=.1, training_steps=100,
               batch_size=1000, correlation_loss=True,
               model_save_file=model_file)
  audio_prediction2 = EvalDNNRegression(all_eeg, model_file, num_hidden=hidden, 
                                        correlation_loss=True)
  print type(audio_prediction2)
  print "Variance of predictions is:", np.var(audio_prediction2)
  print 
  
def regression_main(argv):
  # TestPolynomialFitting()
  # RunDemoDataTest(hidden_number = 3, num_trials=3)
  RunSaveModelTest()

if __name__ == "__main__":
  # Just run a test program
  regression_main(sys.argv)