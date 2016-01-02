% Code to test the performance of the Telluride Decoding Toolbox on 
% Jonathan Simon's MEG Data.

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

[wav_intensity, eeg_data] = PrepareTellurideData('../data/tdt_male_alone.mat');

nTrials = size(eeg_data,3);
%%
% Train on trial 1, test on trial 2... just a simple sanity test.
maxLags = 0:25;
correlations_vs_lag = 0*maxLags;

for mi = 1:length(maxLags)
    lagList = [0:maxLags(mi)];
    [g,pred] = FindTRF(wav_intensity, eeg_data(:,:,1), -1, eeg_data(:,:,2), ...
        [], lagList, 'Shrinkage', 1e-3);

    r = corrcoef(wav_intensity, pred);
    correlations_vs_lag(mi) = r(2,1);
end
%%
clf;
plot(maxLags, correlations_vs_lag);
xlabel('Number of Additional Context Frames (64 Hz)');
ylabel('Correlation');
title('Reconstruction Accuracy vs. Model Length');
%%
% A full test... train on two trials, test on the third.  Do this for a
% range of maxLags to make sure we're getting believable results.
maxLags = 0:2:50;
correlations_vs_lags = 0*maxLags;

for ci=1:length(maxLags)
    
    lagList = 0:ci;
    clear trial_model
    % Create the models for each trial (independently)
    for trial = 1:nTrials
        [g] = FindTRF(wav_intensity, ...
                        eeg_data(:,:,trial), ...
                        -1, [], [], lagList, 'Shrinkage', 1e-3);
        trial_model{trial} = g;
    end
    
    all_rs = [];
    for test_trial = 1:nTrials
        train_trials = setdiff(1:nTrials, test_trial);
        trial_g = 0;
        for train_trial=train_trials;
            trial_g = trial_g + trial_model{train_trial};
        end
        [g, pred] = FindTRF([], [], -1, ...
            eeg_data(:,:,test_trial), trial_g/length(train_trials), ...
            lagList, 'Shrinkage', 1e-3);
        r = corrcoef(wav_intensity, pred);
        all_rs = [all_rs r(2,1)];
    end
    correlations_vs_lags(ci) = mean(all_rs);

    plot(maxLags, correlations_vs_lags);
    xlabel('Number of additional lags in model (f_s=64)');
    ylabel('Correlation');
    title('MEG Model Accuracy (train on 2 trials, test on 3rd)');
    drawnow;
end

%%
% Train and test on the same trial.  Use all but one batch as training
% data, and use that one batch for testing.
lagList = 0:25;
numBatches = 10;
numTrials = 3;
doScale = 1;
shrinkage = 1e-1;
all_rs = zeros(numBatches, numTrials);
for trial_num = 1:numTrials
    for testBatch = 1:numBatches
        trainBatch = ones(1,numBatches);
        trainBatch(testBatch) = 0;
        numFrames = size(eeg_data, 1);
        trainingFrames = true(num_frames, 1);
        begTest = max(1,floor((testBatch-1)/numBatches*numFrames));
        endTest = min(num_frames,floor((testBatch)/numBatches*numFrames));
        trainingFrames(begTest:endTest) = false;
        [g] = FindTRF(wav_intensity, eeg_data(:,:,trial_num), ...
                    -1, [], [], lagList, 'Shrinkage', 1e-1, doScale, ...
                    trainingFrames);
        [g, pred] = FindTRF([], [], -1, ...
                eeg_data(~trainingFrames,:,trial_num), g, ...
                lagList, 'Shrinkage', shrinkage, doScale);
        r = corrcoef(wav_intensity(~trainingFrames), pred);
        all_rs(testBatch, trial_num) = r(2,1);
    end
end
mean(mean(all_rs))
xlabel('Test Batch')
ylabel('Prediction Correlation');
title('Male MEG Predictions');
legend('Trial 1', 'Trial 2', 'Trial 3')


%%
% Build a model for all the male-alone data
[male_wav_intensity, eeg_data] = PrepareTellurideData('../data/tdt_male_alone.mat');

% Build all the models
lagList = 0:25;
male_model = 0;
% Create the models for each trial (independently)
for trial = 1:nTrials
    [g] = FindTRF(male_wav_intensity, ...
                    eeg_data(:,:,trial), ...
                    -1, [], [], lagList, 'Shrinkage', 1e-3);
    male_model = male_model + g;
end
male_model = male_model/nTrials;

%%
% Build a model for all the female-alone data
[female_wav_intensity, eeg_data] = PrepareTellurideData('../data/tdt_female_alone.mat');

% Build all the models
lagList = 0:25;
female_model = 0;
% Create the models for each trial (independently)
for trial = 1:nTrials
    [g] = FindTRF(female_wav_intensity, ...
                    eeg_data(:,:,trial), ...
                    -1, [], [], lagList, 'Shrinkage', 1e-3);
    female_model = female_model + g;
end
female_model = female_model/nTrials;

%%
% Now test both decoders on the male-female mixture, the male speaker is
% the target.
[wav_intensity, eeg_data] = PrepareTellurideData('../data/tdt_male_target.mat');

male_decode_rs = zeros(2, size(eeg_data,3));

for trial_num=1:size(eeg_data,3)
    [~, pred] = FindTRF([], [], -1, ...
            eeg_data(:,:,trial_num), male_model, ...
            lagList, 'Shrinkage', shrinkage, doScale);
    r = corrcoef(male_wav_intensity, pred);
    male_decode_rs(1, trial_num) = r(2,1);
    
    [~, pred] = FindTRF([], [], -1, ...
            eeg_data(:,:,trial_num), female_model, ...
            lagList, 'Shrinkage', shrinkage, doScale);
    r = corrcoef(female_wav_intensity, pred);
    male_decode_rs(2, trial_num) = r(2,1);
end
%%
% Now test both decoders on the male-female mixture, the female speaker is
% the target.
[wav_intensity, eeg_data] = PrepareTellurideData('../data/tdt_female_target.mat');

female_decode_rs = zeros(2, size(eeg_data,3));

for trial_num=1:size(eeg_data,3)
    [~, pred] = FindTRF([], [], -1, ...
            eeg_data(:,:,trial_num), male_model, ...
            lagList, 'Shrinkage', shrinkage, doScale);
    r = corrcoef(male_wav_intensity, pred);
    female_decode_rs(1, trial_num) = r(2,1);
    
    [~, pred] = FindTRF([], [], -1, ...
            eeg_data(:,:,trial_num), female_model, ...
            lagList, 'Shrinkage', shrinkage, doScale);
    r = corrcoef(female_wav_intensity, pred);
    female_decode_rs(2, trial_num) = r(2,1);

end
