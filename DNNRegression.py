'''
Created on Jul 1, 2015

@author: peter
'''
import sys
import getopt
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import data_loader
import cPickle as pkl
from pylearn2.scripts import train as pylearn2train
from pylearn2.utils import serial
from pylearn2.datasets import cache
from theano import tensor as theanoTensor
from theano import function as theanoFunction


class DNNregression:
    def __init__(self, train_network, yaml_path, stimulus_data_path, 
                 response_data_path, weight_path, visualize, context_length, 
                 direction, valid_data_entries_path, forking, verbosity, 
                 debug):
        self.train_network = train_network
        self.yaml_path = yaml_path
        self.stimulus_data_path = stimulus_data_path
        self.response_data_path = response_data_path
        self.visualize = visualize
        self.context_length = context_length
        self.direction = direction
        self.valid_data_entries_path = valid_data_entries_path
        self.forking = forking
        self.verbosity = verbosity
        self.debug = debug
        self.model_path = "./network_best.pkl"
        self.weight_path= weight_path
        if weight_path[-4:] != ".pkl":
            self.model = None
        else:
            self.model = weight_path
    

    
    def train(self):
        """
        See module-level docstring of /pylearn2/scripts/train.py for a more details.
        """
        print 'train network'
        pylearn2train.train(self.yaml_path)
    

    def save_weights(self):
        """
        Saves all weights in a .txt, .npy or .mat file depending on the ending of the 'weight_path'.
        If the path ends in .pkl, the entire model is stored. 
        """
    
        model = serial.load(self.model_path)
    
        weight_dict = {}
    
        for layer in model.layers:
            try:
                weight_dict[layer.layer_name] = layer.get_weights()
            except:
                layer_weights = layer.get_weights_topo()
                weight_dict[layer.layer_name] = layer_weights# without reshaping since it the input/output vector would need to reshaped in the same way which might lead to problems

        if self.weight_path[-4:] == '.pkl':
            print 'saving model ', self.weight_path
            serial.save(self.weight_path, model)
        elif self.weight_path[-4:] == '.mat':
            scipy.io.savemat(self.weight_path[:-4]+'.mat', weight_dict)
        elif self.weight_path[-4:] == '.npy':
            np.save(self.weight_path[:-4], weight_dict)
        else:
            raise Exception('Only ".mat", ".pkl" and ".npy" files are supported as data formats.')
            
    
    def get_predictions(self, prediction_type="regression"):
            
        if self.model == None:
            print 'no evaluation possible since no model was provided.'
            return
            
        try:
            model = serial.load(self.model)
        except Exception as e:
            print("error loading {}:".format(self.model))
            print(e)
            raise Exception("error loading {}:".format(self.model))
     
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop(X)
     
        if prediction_type == "classification":
            Y = theanoTensor.argmax(Y, axis=1)
        else:
            assert prediction_type == "regression"
     
        f = theanoFunction([X], Y, allow_input_downcast=True)
     
        print("loading data and predicting...")
     
        data = data_loader.Data('test')
        if self.direction == 'reverse':
            input_data = data.response_data
            output_path = self.stimulus_data_path
            m, r, c = input_data.shape
            input_data = input_data.reshape((m, r, c , 1))
        elif self.direction == 'forward':
            input_data = data.stimulus_data
            output_path = self.response_data_path
        else:
            raise Exception('Specify either "reverse" or "forward" as direction.')
     
        prediction = f(input_data)
     
        print("writing predictions...")
     
        if output_path[-4:] == '.mat':
            scipy.io.savemat(output_path, {'data': prediction})
        elif output_path[-4:] == '.txt':
            np.savetxt(output_path, prediction)
        elif output_path[-4:] == '.npy':
            np.save(output_path, prediction)
        else:
            raise Exception('Only ".mat", ".txt" and ".npy" files are supported as data formats.')
        
        
        #------------------------------------------------------------------------------ 
        # for testing
        #------------------------------------------------------------------------------ 
        if self.debug:
            try:
                test_output = np.squeeze(data_loader.load_data_from_file('testUnattendedAudioOrg.mat'))
                test_predictions = np.squeeze(data_loader.load_data_from_file(self.stimulus_data_path))
                print 'DEBUG: test_predictions.shape', test_predictions.shape, 'test_output.shape', test_output.shape
                print 'DEBUG: test_predictions.max', np.max(test_predictions), np.argmax(test_predictions)
                print 'DEBUG: ', np.corrcoef(test_predictions, test_output)
            except:
                print 'Detailed correlation analysis only possible for the hello world example.'
                pass
        



    def visulize_results(self):
        if self.direction == 'reverse':
            upper_title = 'Predicted Stimulus'
            lower_title = 'Response'
        else:
            upper_title = 'Stimulus'
            lower_title = 'Predicted Response'
            
        plt.subplot(2,1,1)
        plt.title(upper_title)
        plt.plot(self.stimulus_data_path)
        plt.subplot(2,1,2)
        plt.title(lower_title)
        plt.plot(self.response_data_path)
        plt.savefig(self.visualize)


    
#     def show_weights(self, model_path, out):
#         """
#         Show or save weights to an image for a pickled model
#     
#         Parameters
#         ----------
#         model_path : str
#             Path of the model pkl to show weights for
#         border : bool, optional
#         out : str
#             Output file to save weights to
#         """
#         pv = get_weights_report.get_weights_report(model_path=model_path,
#                                                    rescale="individual")
#     
#         if out is not None:
#             pv.save(out)
        

    
def main():
    #------------------------------------------------------------------------------ 
    # set parameters
    #------------------------------------------------------------------------------ 
    train_network = None
    yaml = None
    stimulus_data = None
    response_data = None
    weight_path = None
    visualize = False
    context_length = 25
    direction = 'reverse'
    valid_data_entries = -1 # this encodes the default to use all entries
    forking = False
    verbosity = 1
    debug = 0
    num_training_epochs = 100
    num_neurons = 5
        
    #------------------------------------------------------------------------------ 
    # parse arguments
    #------------------------------------------------------------------------------ 
    def usage():
        print 'Call python DNNRegression.py with the following arguments: \n', \
                '{--train, --predict} \n', \
                '-m model.yaml \n', \
                '-s stimulus_data  [i/o, default text, or .mat file] \n', \
                '-r response_data [i/o, default text, or .mat file] \n', \
                '-w weights [i/o, default text, or .pkl] \n', \
                '[--visual visualize.png] \n', \
                '[--context N  for N>=1   *25] \n', \
                '[--numEpochs N default is 100]', \
                '[--numNeurons N default is 5]', \
                '[--dir forward/reverse*] \n', \
                '[--valid  which parts are valid, in case of concatenating trials, default all valid.] \n', \
                '[--forking for matlab sake] \n'
                
                
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 
                                "htpm:s:r:w:", ["help", "train", "predict", "model=", 
                                                "stimulus=", "response=", "weights=", 
                                                "visual=", "context=", "dir=", "valid=", 
                                                "forking=", "verbosity=", "debug", 
                                                "numEpochs=", "numNeurons=", "numChannels="])
    except getopt.GetoptError as err:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-p", "--predict"):
            train_network = 0
        elif opt in ("-t", "--train"):
            train_network = 1
        elif opt in ("-m", "--model"):
            yaml = arg
        elif opt in ("-s", "--stimulus"):
            stimulus_data = arg
        elif opt in ("-r", "--response"):
            response_data = arg
        elif opt in ("-w", "--weights"):
            weight_path = arg
        elif opt in ("--visual"):
            visualize = arg
        elif opt in ("--context"):
            context_length = int(arg)
        elif opt in ("--numEpochs"):
            num_training_epochs = int(arg)
        elif opt in ("--numNeurons"):
            num_neurons = int(arg)
        elif opt in ("--dir"):
            direction = arg
        elif opt in ("--valid"):
            valid_data_entries = arg
        elif opt in ("--forking"):
            print "forking is not supported for now"
#             forking = True
        elif opt in ("--verbosity"):
            verbosity = arg
        elif opt in ("--debug"):
            debug = 1
        else:
            assert False, "unhandled option"
            
    if train_network is None:
        print 'You need to define wheter to predict outputs or to train the network.'
        usage()
        sys.exit(2)
    if yaml is None:
        print 'You need to define a valid model path / yaml file.'
        usage()
        sys.exit(2)
    if stimulus_data is None:
        print 'You need to define a valid stimulus data path.'
        usage()
        sys.exit(2)
    if response_data is None:
        print 'You need to define a valid response data path.'
        usage()
        sys.exit(2)
    if weight_path is None:
        print 'You need to define a valid path containing the weights of the mode.'
        usage()
        sys.exit(2)
    
    
    #------------------------------------------------------------------------------ 
    # Save environment for other parts of the DNN
    #------------------------------------------------------------------------------ 
    os.environ["EEGTOOLS_TRAIN_NETWORK"] = str(train_network)
    os.environ["EEGTOOLS_STIMULUS_DATA_PATH"] = stimulus_data
    os.environ["EEGTOOLS_RESPONSE_DATA_PATH"] = response_data
    os.environ["EEGTOOLS_VALID_DATA_PATH"] = str(valid_data_entries)
    os.environ["EEGTOOLS_CONTEXT_LENGTH"] = str(context_length)
    os.environ["EEGTOOLS_DIRECTION"] = direction
    os.environ["EEGTOOLS_DEBUG"] = str(debug)
    
    # determine number of channels from data
    if direction == 'forward':
        input_path = stimulus_data
    else:
        input_path = response_data
    input_data = data_loader.load_data_from_file(input_path)
    num_channels = input_data.shape[1]
    os.environ["EEGTOOLS_NUM_CHANNELS"] = str(num_channels)
    
    #------------------------------------------------------------------------------ 
    # Save imports for the YAML file
    #------------------------------------------------------------------------------ 
    pkl.dump(context_length, open( 'context_length.pkl', 'wb'))
    pkl.dump(num_training_epochs, open( 'num_training_epochs.pkl', 'wb'))
    pkl.dump(num_neurons, open( 'num_neurons.pkl', 'wb'))
    pkl.dump(num_channels, open( 'num_channels.pkl', 'wb'))
    
    #------------------------------------------------------------------------------ 
    # train / run DNN
    #------------------------------------------------------------------------------ 
    net = DNNregression(train_network, yaml, stimulus_data, response_data, 
                 weight_path, visualize, context_length, direction,
                 valid_data_entries, forking, verbosity, debug)
    
    if net.train_network:
        net.train()
        net.save_weights()
    else:
        net.get_predictions()
        
    if net.visualize:
        net.show_weights()

if __name__ == "__main__":
    main()
    
    
    
    
