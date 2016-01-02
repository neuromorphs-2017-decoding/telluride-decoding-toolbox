%%
load ../data/Telluride2015-demo.mat

%%
% Downsample the waveform data to 64Hz.
down_wav = cell(length(data.wav),1);
oldFs = data.fsample.wav; newFs = 64;
for i=1:length(data.wav)
    down_wav{i} = resample_fft(data.wav{i}, oldFs, newFs);
end
%%
% Downsample the EEG data to 64Hz.
down_eeg = cell(length(data.eeg),1);
oldFs = data.fsample.eeg; newFs = 64;
for i=1:length(data.eeg)
    down_eeg{i} = resample_fft(data.eeg{i}, oldFs, newFs);
end

%%
trainStimulusDataFile = [tempname() '.mat'];
data = down_wav{1};
save(trainStimulusDataFile, 'data');

trainResponseDataFile = [tempname() '.mat'];
data = down_eeg{1};
save(trainResponseDataFile, 'data');
clear data

testStimulusDataFile = [tempname() '.mat'];
data = down_wav{2};
save(testStimulusDataFile, 'data');

testResponseDataFile = [tempname() '.mat'];
data = down_eeg{2};
save(testResponseDataFile, 'data');
clear data

fprintf('TrainStimulusDataFile="%s"\n', trainStimulusDataFile);
fprintf('TrainResponseDataFile="%s"\n', trainResponseDataFile);
fprintf('TestStimulusDataFile="%s"\n', testStimulusDataFile);
fprintf('TestResponseDataFile="%s"\n', testResponseDataFile);

%%
