"""
EEG data loader.
"""
__authors__ = "Peter U. Diehl"
__email__ = "peter.u.diehl@gmail.com"

import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import cache
from pylearn2.utils import serial
import os
import scipy.io


def load_data_from_file(path):
    datasetCache = cache.datasetCache
    path_eeg = datasetCache.cache_file(path)
    if path_eeg[-4:] == '.mat':
        filename = path.split('/')[-1][:-4]
        data= scipy.io.loadmat(path)['data']
    elif path_eeg[-4:] == '.txt':
        data = np.loadtxt(path)
    elif path_eeg[-4:] == '.npy':
        data = np.load(path)
    else:
        raise Exception('Only ".mat", ".txt" and ".npy" files are supported as data formats.')
    return data


class Data(dense_design_matrix.DenseDesignMatrix):
    """
    The EEG-single speaker dataset

    Parameters
    ----------
    which_set : str
                Accepted are "train" or "test".
    axes : Defines the input format. The default is time x channel
    """

    def __init__(self, which_set, axes=['b', 0, 1, 'c']):

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])
        

        if which_set not in ['train', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '. Valid values are ["train", "test"].')
        
        self.train_network = int(os.environ['EEGTOOLS_TRAIN_NETWORK'])
        self.stimulus_data_path = os.environ['EEGTOOLS_STIMULUS_DATA_PATH']
        self.response_data_path = os.environ['EEGTOOLS_RESPONSE_DATA_PATH']
        self.valid_data_entries_path = os.environ['EEGTOOLS_VALID_DATA_PATH']
        self.context_length = int(os.environ['EEGTOOLS_CONTEXT_LENGTH'])
        self.direction = os.environ['EEGTOOLS_DIRECTION']
        self.debug = int(os.environ['EEGTOOLS_DEBUG'])
        assert self.direction == 'forward' or self.direction == 'reverse'

        
        
        if self.train_network:
            self.stimulus_data = load_data_from_file(self.stimulus_data_path)
            self.response_data = load_data_from_file(self.response_data_path)
            if self.valid_data_entries_path != '-1':
                self.valid_data_entries = np.squeeze(load_data_from_file(self.valid_data_entries_path))
            else:
                if self.direction == 'forward':
                    self.valid_data_entries = np.ones((self.stimulus_data.shape[0],))
                else:
                    self.valid_data_entries = np.ones((self.response_data.shape[0],))
                    
            
            self.stimulus_data, self.response_data = self.slice_data()
            
            m, r, c = self.response_data.shape
            self.response_data = self.response_data.reshape(m, r, c, 1)
            self.stimulus_data = self.stimulus_data.reshape(m, self.stimulus_data.shape[1])
            if np.isnan(self.response_data).any():
                raise Exception('Found NaN in the response data.')
            if np.isnan(self.stimulus_data).any():
                raise Exception('Found NaN in the stimulus data.')
    
            super(Data, self).__init__(topo_view=dimshuffle(self.response_data), 
                                       y=self.stimulus_data.astype(np.float32),
                                        axes=axes)
        elif self.direction == 'forward':
            self.stimulus_data = load_data_from_file(self.stimulus_data_path)
            if self.valid_data_entries_path != '-1':
                self.valid_data_entries = np.squeeze(load_data_from_file(self.valid_data_entries_path))
            else:
                self.valid_data_entries = np.ones((self.stimulus_data.shape[0],))
            self.stimulus_data, _ = self.slice_data()
        elif self.direction == 'reverse':
            self.response_data = load_data_from_file(self.response_data_path)
            if self.valid_data_entries_path != '-1':
                self.valid_data_entries = np.squeeze(load_data_from_file(self.valid_data_entries_path))
            else:
                self.valid_data_entries = np.ones((self.response_data.shape[0],))
            _, self.response_data = self.slice_data()
        else:
            raise Exception('Specify either "reverse" or "forward" as direction.')
        


    


    def slice_data(self):
        num_timesteps = len(self.valid_data_entries)
        num_channels = self.response_data.shape[1]
        defined_num_channels = int(os.environ["EEGTOOLS_NUM_CHANNELS"])
        assert num_channels == defined_num_channels
        num_samples = sum(self.valid_data_entries)
        
        if self.debug:
            print 'self.valid_data_entries', self.valid_data_entries, 'np.squeeze(self.valid_data_entries)', np.squeeze(self.valid_data_entries)
            print 'defined_num_channels', defined_num_channels, 'num_channels', num_channels
            print 'num_samples', num_samples, 'num_timesteps', num_timesteps
            if self.train_network or self.direction == 'reverse':
                print 'num_channels', num_channels, 'response_data.shape', self.response_data.shape
            if self.train_network or self.direction == 'forward':
                print 'stimulus_data.shape', self.stimulus_data.shape
                
        if self.train_network or self.direction == 'forward':
            stimulus_samples = np.zeros((num_samples-self.context_length, 1))
        if self.train_network or self.direction == 'reverse':
            response_samples = np.zeros((num_samples-self.context_length, self.context_length, num_channels))
        
        
        assert self.context_length > 0 
        assert num_timesteps > self.context_length
        assert self.response_data.shape[0] > self.context_length
            

        current_sample = 0
        
        for i in xrange(num_timesteps-self.context_length):
            if np.all(self.valid_data_entries[i:i+self.context_length]):   #TODO: make slicing aware of ANY invalid entries, not just the first one
                if self.direction == 'forward':
                    if self.train_network:
                        response_samples[current_sample,:,:] = np.squeeze(self.response_data[i, :])
                    stimulus_samples[current_sample] = self.stimulus_data[i : i+self.context_length]
                if self.direction == 'reverse':
                    response_samples[current_sample,:,:] = np.squeeze(self.response_data[i : i+self.context_length, :])
                    if self.train_network:
                        stimulus_samples[current_sample] = self.stimulus_data[i]
                current_sample += 1
                    
                    
        if self.train_network:
            return stimulus_samples, response_samples
        elif self.direction == 'forward':
            return stimulus_samples, None
        else:
            return None, response_samples
    
