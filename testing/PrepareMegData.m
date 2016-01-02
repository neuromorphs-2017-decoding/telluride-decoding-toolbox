function [wav_intensity, eeg_data] = PrepareMEGData(filename, newFs, derivative)
% function [wav_intensity, eeg_data] = PrepareMEGData(filename, newFs, 
%	derivative)
% Read in data from the Jonathan Simon's MEG dataset and prepare it for 
% the TestMegData routines.

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

if nargin < 2 || isempty(newFs) || newFs < 0
    newFs = 64;
end

if nargin < 3
    derivative = 0;
end

load(filename)

nTrials = size(data.eeg{1},3);

% Compute the intensity feature (downsample too)
if isfield(data.fsample, 'wav')
    origFs = data.fsample.wav;
elseif isfield(data.fsample, 'audio')
    origFs = data.fsample.audio;
end
wav_intensity = CreateLoudnessFeature(data.wav{1}, origFs, newFs);

% Compute the intensity derivative
if derivative
    wav_intensity = [wav_intensity(2:end); wav_intensity(end)] - ...
                    wav_intensity;
    wav_intensity = max(0, wav_intensity);
end

% Downsample and create the corresponding eeg data.
clear eeg_data
for i=1:nTrials
    new_eeg = resample_fft(data.eeg{1}(:,:,i), data.fsample.eeg, newFs);
    eeg_data(:,:,i) = new_eeg(65:end-64,:);
end
