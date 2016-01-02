% Code to implement decoding using Google's Tensorflow machine-learning system.
% This version only uses a single layer with a rectified linear unit (RELU) but
% does not (yet) do a good job decoding EEG data.

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

import math, random, sys
import matplotlib.pyplot as plot
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

# Create and use a simple Tensorflow network to predict the (EEG) response from
# a stimulus (e.g. audio intensity). '''
class RegressionNetwork:
  kStimulusChannels = 1       # Only one channel of input
  
  def __init__(self, channel_count, half_width = 13, single_sided = True):
    '''Initialize the network with the number of response channels, the size 
    of the filter to predict, and whether we want a single-sided or double-sided
    in time prediction.'''
    self.channel_count = channel_count    # Response channels
    self.half_width = half_width          # Width of filter on each side of zero
    self.single_sided = single_sided      # One or two sided filter (+/- time)
    # Now the internal variables
    self.tResponse = None     # Placeholder for response signal
    self.tStimulus = None     # Placeholder for stimulus signal
    self.tW = None            # TF matrix with the regression matrix
    self.tTrain = None        # TF object used to train the network
    self.tLinear = None       # TF object before RELU
    self.tPrediction = None   # TF output variable
    self.tInit = None         # TF initialization object
    self.tLoss = None         # TF loss object
    # Running variables
    self.session = None
    self.print_loss_interval = 20 # How often to print the network loss
    self.learning_rate = 1e-3
    
  def Create(self):
    '''Create a Tensorflow convolutional network that predicts stimulus from the
    response.'''
    self.tResponse = tf.placeholder(tf.float32, 
      shape=[1, self.channel_count, None, 1,], 
    	name='response')
    self.tStimulus = tf.placeholder(tf.float32, 
      shape=[1, RegressionNetwork.kStimulusChannels, None, 1], 
    	name='stimulus_prediction')
    
    if self.single_sided:
      w_shape = [self.channel_count,self.half_width,1,1]
      self.tW = tf.Variable(tf.random_uniform(w_shape, -1.0, 1.0),
    	  name='ReconstructionFilter')
    else:
      w_shape = [self.channel_count,2*self.half_width+1,1,1]
      self.tW = tf.Variable(tf.random_uniform(w_shape, -1.0, 1.0),
    	  name='ReconstructionFilter')
    self.tLinear = tf.nn.conv2d(self.tResponse, self.tW, (1,1,1,1), 
      "VALID", name='conv_output')
    self.tPrediction = tf.nn.relu(self.tLinear, name="RELU")
    
    # Minimize the mean squared errors.
    error = tf.square(self.tPrediction - self.tStimulus)
    self.tLoss = tf.reduce_mean(error)
    if 0:
      # Fewer iterations needed for Test().
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    else:
      # This seems to be more reliable for true data (without need for scaling)
      # http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 
        self.learning_rate/2)
    self.tTrain = optimizer.minimize(self.tLoss)
    
    # Before starting, initialize the variables.  We will 'run' this first.
    self.tInit = tf.initialize_all_variables()
  
  def Optimize(self, TrainingStim, TrainingResp, iterations=200, batch_size=100):
    '''Optmize the network given the stimulus and the response.  In both cases
    the data is N_times x N_channels numpy array.  There is 1 stimulus channel,  
    and self.channel_count response channels. '''
    if not isinstance(TrainingStim, np.ndarray):
      raise TypeError("Stimulus must be an numpy array")
    if TrainingStim.shape[1] != 1:
      raise ValueError("Stimulus must be Nx1 in size")
    num_train_frames = TrainingStim.shape[0]
    if not isinstance(TrainingResp, np.ndarray):
      raise TypeError("Training response must be an numpy array")
    if TrainingResp.shape[0] != num_train_frames:
      raise ValueError("Stimulus and response must both have same # rows")
    if TrainingResp.shape[1] != self.channel_count:
      raise ValueError("Response must have %d rows" % self.channel_count)
    # Launch the graph.
    if self.session == None:
      self.session = tf.Session()
      self.session.run(self.tInit)
    TrainingResp = TrainingResp.T
    TrainingStim = TrainingStim.T
    if self.single_sided:
      delta =- self.half_width + 1
      for iter_num in xrange(iterations):
        # Pick a random piece of the training data for this minibatch.
        start = random.randint(self.half_width+delta, num_train_frames-batch_size)
        end = start + batch_size
        resp_feed = np.reshape(TrainingResp[:,start:end],
                               (1,self.channel_count,-1,1))
        # First half_width/2 input examples are dropped, but that's ok, 
        # because we want the first output sample to line up.  
        # So actual start is 0.
        stim_piece = TrainingStim[:,start+self.half_width-1+delta:end+delta]
        stim_feed = np.reshape(stim_piece, (1,1,-1,1))
        [loss_value,train_value,conv_filter,prediction_value] = \
          self.session.run([self.tLoss,self.tTrain,self.tW,self.tPrediction], 
                            feed_dict={self.tResponse: resp_feed, 
      			                           self.tStimulus: stim_feed})
        if self.print_loss_interval > 0 and iter_num % self.print_loss_interval == 0:
          print loss_value
          plot.clf()
          plot.plot(np.hstack((stim_piece.T,prediction_value.reshape((1,-1)).T)))
          plot.title('Step %d' % iter_num)
          plot.legend(('Stimulus','Prediction'))
          plot.gcf().canvas.draw()
          plot.show(block=False)
    else:
      for iter_num in xrange(iterations):
        start = random.randint(self.half_width, num_train_frames-batch_size)
        end = start + batch_size
        resp_feed = np.reshape(TrainingResp[:,start:end],
                                (1,self.channel_count,-1,1))
        # Because of VALID in convolution, output is shrunk by half_width on both
        # ends
        stim_piece = TrainingStim[:,start+self.half_width:end-self.half_width]
        stim_feed = np.reshape(stim_piece, (1,1,-1,1))
        [loss_value,train_value,conv_filter,prediction_value] = \
          self.session.run([self.tLoss,self.tTrain,self.tW,self.tPrediction], 
      					           feed_dict={self.tResponse: resp_feed, 
      						                    self.tStimulus: stim_feed})
        if self.print_loss_interval > 0 and iter_num % self.print_loss_interval == 0:
          print loss_value
          plot.clf()
          plot.plot(np.hstack((stim_piece.T,prediction_value.reshape((1,-1)).T)))
          plot.title('Step %d' % iter_num)
          plot.legend(('Stimulus','Prediction'))
          plot.gcf().canvas.draw()
          plot.show(block=False)
    return (stim_piece, resp_feed,prediction_value)      # For testing
  
  def RetrieveFilter(self):
    '''Retrieve the filter that has been learned.  Returns an numpy array.'''
    if self.session == None:
      return None
    conv_filter = self.session.run(self.tW)
    return conv_filter.squeeze().T
    
  def Predict(self, response):
    '''Given some new response data (in the form of an numpy array), use the 
    model to predict the original stimulus (which is returned as a Nx1 numpy
    array.)'''
    if not isinstance(response, np.ndarray):
      raise TypeError("Training response must be an numpy array")
    if response.shape[1] != self.channel_count:
      raise ValueError("Response must have %d rows" % self.channel_count)
    if self.session == None:
      self.session = tf.Session()
      self.session.run(self.tInit)
    response = response.T
    if self.single_sided:
      delta =- self.half_width + 1
      resp_feed = np.reshape(response, (1,self.channel_count,-1,1))
        # First half_width/2 input examples are dropped, but that's ok, 
        # because we want the first output sample to line up.  
        # So actual start is 0.
      prediction = self.session.run(self.tPrediction, 
                            feed_dict={self.tResponse: resp_feed})
      padding = np.zeros((self.half_width-1, 1))
      padded_prediction = np.vstack((
        prediction.reshape(RegressionNetwork.kStimulusChannels, -1).T,
        padding))
    else:
      resp_feed = np.reshape(response, (1,self.channel_count,-1,1))
      # Because of VALID in convolution, output is shrunk by half_width on both
      # ends
      prediction = self.session.run(self.tPrediction, 
    					           feed_dict={self.tResponse: resp_feed})
      padding = np.zeros((self.half_width, 1))
      padded_prediction = np.vstack((padding, 
        prediction.reshape(RegressionNetwork.kStimulusChannels, -1).T,
        padding))
    return padded_prediction
    
  @staticmethod
  def Test(singleSided = True, iterations=2000, batch_size=200):
    '''Test the reconstruction process with a known signal.  This is a 
    static method so it can allocate the network object, design the network,
    optimize it, and measure the final performance.  Training and testing on 
    the same data, so the correlation values at the end should be nearly 1.'''
    # Generate the good part of the stimulus, 1 channel over time
    N = 10000
    s = np.random.randn(N, 1)
    StimulusData = np.select([s<0, s>=0], [0*s, s])
    nStimulusChannels = 1
    # StimulusData = np.select([StimulusData < 0, StimulusData >= 0], 
    #                          [0, StimulusData])
    # Create the response.  
    # First channel is 0.5 times current time, plus 0.25 times previous time.  
    # Second channel is all noise.
    ResponseData = np.hstack((StimulusData[:,:] * 0.5 +
    			  np.vstack((StimulusData[1:,:], np.zeros((1,1)))) * 0.25,
    			 np.random.randn(N,1)))
    # Do the training/testing split
    kNumTestFrames=1000
    TestingTimes = np.arange(kNumTestFrames) # Grab first 1000 samples for testing
    TrainingTimes = np.arange(kNumTestFrames, N)

    TrainingStim = StimulusData[TrainingTimes,:]
    TrainingResp = ResponseData[TrainingTimes,:]
    TestingStim = StimulusData[TestingTimes,:]
    TestingResp = ResponseData[TestingTimes,:]
    
    kChannelCount = 2				# EEG Convolution Size
    kHalfWidth = 13
    
    regressor = RegressionNetwork(kChannelCount, kHalfWidth, singleSided)
    regressor.Create()
    # regressor.learning_rate = 0.5
    regressor.Optimize(TrainingStim, TrainingResp, \
      iterations=iterations, batch_size=batch_size)
    # Now plot the results.  First channel's filter should oscillate and decay by
    # 50% at each time step. Second channel's filter is zero since it is noise.
    conv_filter = regressor.RetrieveFilter()
    plot.clf()
    if singleSided:
      plot.plot(np.arange(0,kHalfWidth).reshape(-1,1),conv_filter)
    else:
      plot.plot(np.arange(-kHalfWidth, kHalfWidth+1).reshape(-1,1), conv_filter)
    plot.title('Predicted Filter Response')
    plot.xlabel('Time')
    plot.ylabel('Filter Response')
    # Test the predictions
    prediction = regressor.Predict(TestingResp)
    if prediction.shape != TestingStim.shape:
      raise ValueError("Internal Error: prediction and TestingStim " + 
        "must be same size.")
    r = np.corrcoef(TestingStim.T, prediction.T)
    print "Correlation between test stimulus and prediction is ", r[1,0]
    return regressor
    
def RunRegression(TrainStimulusDataFile, TrainResponseDataFile, \
  TestStimulusDataFile=None, TestResponseDataFile=None):
  eegScale = 100
  
  m=loadmat(TrainStimulusDataFile)
  trainStimulusData = m['data']
  
  m = loadmat(TrainResponseDataFile)
  trainResponseData = m['data']/eegScale
  newFs = 64;
  
  if TestStimulusDataFile is not None and TestStimulusDataFile is not "":
    m = loadmat(TestStimulusDataFile)
    testStimulusData = m['data']
  else:
    testStimulusData = None
  if TestResponseDataFile is not None and TestResponseDataFile is not "":
    m = loadmat(TestResponseDataFile)
    testResponseData = m['data']/eegScale
  else:
    testResponseData = None
  #
  channel_count = trainResponseData.shape[1]
  half_width = int(0.25*newFs + 0.9999)
  single_sided = False
  regressor = RegressionNetwork(channel_count, half_width, single_sided)
  # regressor.learning_rate = 0.1
  # regressor.print_loss_interval = 1
  regressor.Create()
  regressor.print_loss_interval = 200
  (stim_piece, resp_feed,prediction_value) = \
  regressor.Optimize(trainStimulusData, trainResponseData, 
                      batch_size=1000, iterations=2000)
  
  if testResponseData == None and testStimulusData == None:
    prediction = regressor.Predict(trainResponseData)
    r = np.corrcoef(trainStimulusData.T, prediction.T)
    dataset = 'training'
  else:
    prediction = regressor.Predict(testResponseData)
    r = np.corrcoef(testStimulusData.T, prediction.T)
    dataset = 'testing'
  print "Correlation between %s stimulus and prediction is %g" % \
    (dataset, r[1,0])
  
    
if __name__ == "__main__":
  print sys.argv, len(sys.argv)
  if len(sys.argv) != 3 and len(sys.argv) != 5:
    print "Syntax: %s TrainStimulusDataFile TrainResponseDataFile " % sys.argv[0],
    print "[TestStimulusDataFile TestResponseDataFile]"
  else:
    TrainStimulusDataFile = sys.argv[1]
    TrainResponseDataFile = sys.argv[2]
    if sys.argv >= 5:
      TestStimulusDataFile = sys.argv[3]
      TestResponseDataFile = sys.argv[4]
    else:
      TestStimulusDataFile = None
      TestResponseDataFile = Nonefg
    RunRegression(TrainStimulusDataFile, TrainResponseDataFile, \
      TestStimulusDataFile, TestResponseDataFile)
      
