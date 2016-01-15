# Telluride Decoding Toolbox.
Sahar Akram (UMD), Alain de Chevigne (ENS), Peter Udo Diehl (ETH), 
Jens Hjortjaer (DTU), Nima Mesgarani (Columbia), Lucas Parra (NYU), 
Shihab Shamma (UMD), Malcolm Slaney (Google), Daniel Wong (ENS)

Full documentation is available at: 
https://docs.google.com/document/d/1PYBzEE_aJ5_DpI2AnYKffa3ieckXuSb0eJUTvL1OVlI/edit?usp=sharing

Home page is at: http://www.ine-web.org/software/decoding/

Our goal is to provide a standard set of tools that allow users to decode 
brain signals into the signals that generated them, 
whether the signals come from visual or auditory stimuli, and 
whether they are measured with EEG, MEG, ECoG or any other 
response that you would like to decode. While the developers of this 
toolbox are largely researchers that meet in Telluride Colorado for a 
Neuromorphic workshop and use EEG to analyze auditory experiments,
the tools in this toolbox allow any perceptual stimulus to be connected 
to any neural signal. 

Classically, temporal responses of EEG signals have been analyzed using 
event-related potentials (ERP) or time-frequency analysis. 
ERP studies use a discrete short signal, average the response over many 
repetitions, and look for changes in the peaks and valleys of the 
stimulus-locked response. MMN (mismatch negativity) is a 
particular event-related response that examines oddball responses in the 
context of repetitive stimuli. 
Time-frequency analysis examines the changes in the EEG spectrum in 
particular bands over time. BCI (brain-computer interface) experiments, 
may look for some spectral change in the brain response that researchers 
can use to indicate a choice—the problem is treated as a classification problem.

This toolbox takes a new approach.
Given a stimulus and the resulting brain response, 
we want to decode the response to find the original stimuli. 
In the simplest case we want to predict the auditory signal that 
generated the measured EEG response, or we might want to go 
further and predict to which of two auditory stimuli a user is attending. 
The goal is to produce a continuous prediction of the neural signals 
that are likely to be produced by a stimulus (the forward problem), 
or given an EEG or MEG response, predict the stimulus that generated it.

This is a hard problem. Given an EEG or MEG signal, we can’t predict the 
auditory signal with high fidelity. 
But we can say with much better than chance whether the subject was 
listening to speaker A or speaker B. 
There is information that can be decoded. 
With better machine-learning technology and more data, 
the time has come to think about neural decoding. 
People have done this with spikes, 
but we want to look at ensembles of signals. 
While the correlations are still low, we think they will get better with time.

Overview
A common application of these tools is to decode attention: 
estimate which auditory source is a listener attending.
As shown in the attention decoding block diagram below, 
this toolbox addresses three parts of the decoding problem: 
predicting the stimuli, correlating with the expected signals, and 
deciding the attended speaker. 



To connect audio and EEG (or the reverse) we provides two 
basic forms of predictions: linear and non-linear. 
They each have different strengths and needs.
Linear prediction is easiest to formulate, analyze, and calculate. 
Non-linear predictions based on deep neural networks (DNNs) 
represent a rich source of possibilities but with a larger computational cost. 

Then to decide which speaker a subject is attending, 
we correlate the predictions with the original stimuli.
Correlation answers a simple question, are two signals related, 
and thus might perform better in some situations. 
This can be done in any of several feature domains, such as intensity, 
MFCC and spectrograms. 

Finally, given the correlation signals, we need to decide to which speaker the subject is attending. This can be done by simply summing the correlations over time, and picking the winner. Or a more principled approach combines a probabilistic model of how long a time a subject is likely to attend to one subject along with the behavior of the correlation signal, and chooses a winner that optimizes a loss function that combines these two factors.

Most importantly, this toolbox calculates temporal response functions, TRF’s. 
The TRF describes the connection between the stimulus and the 
recorded responses, or the inverse, going from responses back to the stimuli. 
For the linear model (FindTRF) this is equivalent to a multi-channel 
convolution. 
While for the DNN approach the TRF is a non-linear function of its inputs.

Decoding algorithms depend on temporal context to make their predictions.
For example when predicting the forward response (audio->eeg), 
the audio right now will generate a response over several tens of 
milliseconds to come. 
Conversely, to estimate the EEG signal now we need to know the audio 
signal from the previous ~200ms. 
This is called context, and is shown in the figure below. 
In these cases, the model predicts the response based on 7 frames of data. 
The FindTRF and DnnRegression routines take a context-window size, 
which is an integer representing the number of frames of data, 
either forward or backward in time depending on the model direction, 
to include as input data in the analysis.


The toolbox includes the code (Matlab or Python) to implement the 
four models described above, along with necessary test code. 
We have also included sample EEG and MEG data so new users can 
verify that the algorithms work as promised.

We demonstrate the basic behavior of the three algorithms with a 
synthetic dataset. 
This dataset starts with a deterministic attention signal and 
creates two random 32 channel EEG system impulse responses, 
one for the attended and one for the unattended signal. 
It correlates two sinusoidal test signals with the appropriate 
impulse response to get the EEG response.
The goal of this toolbox is to reconstruct the original attention signal---
was the subject listening to speaker 1 (sinusoid test signal 1) 
or speaker 2 (sinusoid test signal 2). 
See the CreateDemoSignals function for the setup, 
and each algorithm for its behavior.

