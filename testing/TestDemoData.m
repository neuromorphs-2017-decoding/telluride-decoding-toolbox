% Code to test the performance of the Telluride Decoding Toolbox on the 
% Telluride demo data (four simultaneous EEG recordings while listening to 
% single AV presentation.

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

% Run these tests from the testing directory.  That means the actual 
% are one level up the directory hierarchy.

if ~exist('FindTRF', 'file')
    error('Can''t find the the Telluride Decoding Toolbox routines.');
end

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

nTrials = size(eeg_data,3);

%%
% Now create a model for each trial.
Lags = 0:round(newFs/3);
Method = 'Shrinkage';
model = cell(1,length(down_eeg));
for i=1:length(data.eeg)
    stimulus_number = mod(i-1,4) + 1;
    model{i} = FindTRF(down_wav{stimulus_number}, ...
        down_eeg{i}, -1, [], [], Lags, Method);
end

%%
% Now summarize the models.
allModels = zeros(size(model{1},1), size(model{1},2), length(model));
for i=1:length(model)
    allModels(:,:,i) = model{i};
end
figure(1); clf;
subplot(1,2,1);
imagesc(mean(allModels, 3)'); title('Mean TRF'); colorbar;
ylabel('Channel'); xlabel('Time (sample)')
subplot(1,2,2);
imagesc(std(allModels,0, 3)'); title('Std TRF'); colorbar;
ylabel('Channel'); xlabel('Time (sample)')

%%
% Train and test lots of models.
pred_wav{i} = cell(length(data.eeg), 1);
self_correlation = zeros(1, length(data.eeg));
random_correlation = self_correlation;
meanmodel_correlation = self_correlation;
cleanmodel_correlation = self_correlation;

mean_model = mean(allModels, 3);

for i=1:length(data.eeg)
    % Train on the model we created for this trial (cheat)
    stimulus_number = mod(i-1,4) + 1;
    [~, prediction] = FindTRF([], [], ...
        -1, down_eeg{i}, model{i}, Lags, Method);
    pred_wav{i} = prediction;
    r = corrcoef([down_wav{stimulus_number} prediction]);
    self_correlation(i) = r(2,1);

    % Train on the model from a different trial (clean, but pessimistic)
    ni = mod(i+1, length(data.eeg)) + 1;
    [~, prediction] = FindTRF([], [], ...
        -1, down_eeg{i}, model{ni}, Lags, Method);
    pred_wav{i} = prediction;
    r = corrcoef([down_wav{stimulus_number} prediction]);
    random_correlation(i) = r(2,1);
    
    % Train on the average model based on all the trial data (semi-cheat)
     [~, prediction] = FindTRF([], [], ...
        -1, down_eeg{i}, mean_model, Lags, Method);
    pred_wav{i} = prediction;
    r = corrcoef([down_wav{stimulus_number} prediction]);
    meanmodel_correlation(i) = r(2,1);

   % Train on the average of all the models except this one (clean!!)
    clean_model = allModels;
    clean_model(:,:,i) = [];
    clean_model = mean(clean_model, 3);
    [~, prediction] = FindTRF([], [], ...
        -1, down_eeg{i}, clean_model, Lags, Method);
    % pred_wav{i} = prediction;
    r = corrcoef([down_wav{stimulus_number} prediction]);
    cleanmodel_correlation(i) = r(2,1);    
end

%%

figure(2); clf
plot([self_correlation' meanmodel_correlation' cleanmodel_correlation' random_correlation' ])
legend('Self Correlation', 'Mean Model Correlation', 'Jackknife Model Correlation', 'Random Model Correlation');
xlabel('Trial Number'); ylabel('Correlation between intensity and prediction');
title('Demo Data Model Correlation');

%%
kValues = [.01 0.1 0.2 0.5 1 2 4 8 16 32 64];
methodValues = {'None', 'Shrinkage', 'Ridge', 'NRC', 'CCA', 'DNN'};
regular_results = zeros(length(kValues), length(methodValues));

for mi = 1:length(methodValues)         % By Model
    Method = methodValues{mi};
    if strcmp(Method, 'CCA'); continue; end
    if strcmp(Method, 'DNN'); continue; end
    for ki = 1:length(kValues)          % By regularization
        K = kValues(ki);
        if (strcmp(Method, 'Shrinkage') && K > 1) || ...
           (strcmp(Method, 'NRC') && K > 1) || ...
           (strcmp(Method, 'Lasso') && K > 1)
            regular_results(ki, mi) = NaN;
            continue;
        end
        % Now create a model for each trial.
        Lags = 0:round(newFs/3);
        model = cell(1,length(down_eeg));
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            model{i} = FindTRF(down_wav{stimulus_number}, ...
                down_eeg{i}, -1, [], [], Lags, Method, K);
        end
        allModels = zeros(size(model{1},1), size(model{1},2), length(model));
        for i=1:length(model);
            allModels(:,:,i) = model{i};
        end

        % Train on the average of all the models except this one (clean!!)
        testmodel_correlation = zeros(1,length(data.eeg));
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            test_model = allModels;
            test_model(:,:,i) = [];
            test_model = mean(test_model, 3);
            [~, prediction] = FindTRF([], [], ...
                -1, down_eeg{i}, test_model, Lags, Method);
            % pred_wav{i} = prediction;
            r = corrcoef([down_wav{stimulus_number} prediction]);
            testmodel_correlation(i) = r(2,1);
        end
        regular_results(ki, mi) = mean(testmodel_correlation(isfinite(testmodel_correlation)));
    end
end

%%
% Now do the CCA approach.
allModelsWave = [];
allModelsEEG = [];
for mi = 1:length(methodValues)         % By Model
    Method = methodValues{mi};
    if ~strcmp(Method, 'CCA'); continue; end
    for ki = 1:length(kValues)          % By regularization
        K = kValues(ki);
        
        % Now create a model for each trial.
        Lags = 0:round(newFs/3);
        modelWave = cell(1,length(down_eeg));
        modelEEG = modelWave;
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            laggedTrainResponse = LagGenerator(down_eeg{i}, Lags);
            [wAudio,wEEG] = cca(down_wav{stimulus_number}', laggedTrainResponse', K);
            allModelsWave(:,:,i) = wAudio;
            allModelsEEG(:,:,i) = wEEG;
        end
        
        % Train on the average of all the models except this one (clean!!)
        testmodel_correlation = zeros(1,length(data.eeg));
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            test_model_eeg = allModelsEEG;
            test_model_eeg(:,:,i) = [];
            wAttendedEEG = mean(test_model_eeg, 3);
            test_model_wave = allModelsWave;
            test_model_wave(:,:,i) = [];
            wAttendedWave = mean(test_model_wave, 3);
            
            laggedTestResponse = LagGenerator(down_eeg{i}, Lags);
            ccaEEGPrediction = laggedTestResponse * wAttendedEEG;
            ccaWavePrediction = down_wav{stimulus_number} * wAttendedWave;
            
            r = corrcoef([ccaWavePrediction ccaEEGPrediction]);
            testmodel_correlation(i) = r(2,1);
        end
        regular_results(ki, mi) = mean(testmodel_correlation(isfinite(testmodel_correlation)));
    end
end

%%
% Now do the DNN approach.
allModelsWave = [];
allModelsEEG = [];
testmodel_dnn = [];
for mi = 1:length(methodValues)         % By Model
    Method = methodValues{mi};
    if ~strcmp(Method, 'DNN'); continue; end
    
    Lags = round(newFs/3);
    maxFrames = 0;
    for i=1:length(down_eeg)        % Find the longest, we'll shorten later
        maxFrames = maxFrames + size(down_eeg{i},1);
    end
    % Now create a separate model for all trials
    for testTrial = 1:length(down_eeg)
        numTrainSamples = maxFrames - size(down_eeg{testTrial},1);
        allEEG = zeros(numTrainSamples, ...
            size(down_eeg{1},2));
        allWave = zeros(numTrainSamples, size(down_wav{1},2));
        validTrials = ones(numTrainSamples, 1);
        trainTrials = setdiff(1:length(down_eeg), testTrial);
        i = 1;
        % Create the training data for this trial (excluding testTrial)
        for ti = trainTrials
            numFramesThisTrial = size(down_eeg{ti},1);
            allEEG(i:i+numFramesThisTrial-1, :) = down_eeg{ti};
            stimulus_number = mod(ti-1,4) + 1;
            allWave(i:i+numFramesThisTrial-1, :) = down_wav{stimulus_number};
            i = i + numFramesThisTrial;
            validTrials(i,1) = 0;
        end
        % Shorten the results
        allEEG = allEEG(1:i-1,:);
        allWave = allWave(1:i-1,:);
        
        verbosity = 1;
        [g,pred] = DNNRegression(allWave, allEEG, -1, ...
            down_eeg{testTrial}, [], Lags, validTrials, verbosity);
       	stimulus_number = mod(testTrial-1,4) + 1;
        nSamples = min(size(down_wav{stimulus_number}, 1),size(pred, 1));
        r = corrcoef([down_wav{stimulus_number}(1:nSamples,:) pred(1:nSamples,1)]);
        testmodel_dnn(testTrial) = r(2,1);
    end
    regular_results(:,mi) = mean(testmodel_dnn);
end

%%
figure(3); clf;
methodLegend = methodValues;    % Rewrite labels to make them readable
methodLegend{1} = 'No Regularization';
for i=1:4
    methodLegend{i} = ['Linear - ' methodLegend{i}];
end
semilogx(kValues, regular_results);
xlabel('Regularization Value (K)');
ylabel('Correlation');
legend(methodLegend, 'Location', 'Best');
title('Prediction Performance with Different Regularizers');
%%
kRepresentationValues = [.01 0.1 0.2 0.5];
Method = 'shrinkage';
audioRepresentations = {'Intensity', 'Derivative', 'Onset', 'Offset'};
representation_results = zeros(length(kRepresentationValues), ...
                               length(audioRepresentations));

for mi = 1:length(audioRepresentations)         % By representation
    representation = audioRepresentations{mi};
    clear wav_stim;
    for i=1:length(data.wav)
        switch lower(representation)
            case 'intensity'
                wav_stim{i} = down_wav{i};
            case 'derivative'
                wav_stim{i} = [down_wav{i}(2:end); ...
                            down_wav{i}(end)] - ...
                    down_wav{i};
            case 'onset'
                wave_derivative = ...
                    [down_wav{i}(2:end); ...
                     down_wav{i}(end)] - ...
                    down_wav{i};
                wav_stim{i} = max(0, wave_derivative);
            case 'offset'
                wave_derivative = ...
                    [down_wav{i}(2:end); ...
                     down_wav{i}(end)] - ...
                    down_wav{i};
                wav_stim{i} = min(0, wave_derivative);
            case 'binary intensity'
                wav_stim{i} = 1.0*(down_wav{i} > 0);
            case 'binary derivative'
                wave_derivative = ...
                    [down_wav{i}(2:end); ...
                     down_wav{i}(end)] - ...
                    down_wav{i};
                wav_stim{i} = 1.0*(wave_derivative > 0);
            otherwise
                disp('Unknown audio representation');
        end
    end
    for ki = 1:length(kRepresentationValues)          % By regularization
        K = kRepresentationValues(ki);
        % Now create a model for each trial.
        Lags = 0:round(newFs/3);
        model = cell(1,length(down_eeg));
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            model{i} = FindTRF(wav_stim{stimulus_number}, ...
                down_eeg{i}, -1, [], [], Lags, Method, K);
        end
        allModels = zeros(size(model{1},1), size(model{1},2), length(model));
        for i=1:length(model);
            allModels(:,:,i) = model{i};
        end

        % Train on the average of all the models except this one (clean!!)
        testmodel_correlation = zeros(1,length(data.eeg));
        for i=1:length(data.eeg)
            stimulus_number = mod(i-1,4) + 1;
            test_model = allModels;
            test_model(:,:,i) = [];
            test_model = mean(test_model, 3);
            [~, prediction] = FindTRF([], [], ...
                -1, down_eeg{i}, test_model, Lags, Method);
            % pred_wav{i} = prediction;
            r = corrcoef([wav_stim{stimulus_number} prediction]);
            testmodel_correlation(i) = r(2,1);
        end
        representation_results(ki, mi) = mean(testmodel_correlation(isfinite(testmodel_correlation)));
    end
end
%%
figure(4);

semilogx(kRepresentationValues, representation_results);
xlabel('Regularization Value (K)');
ylabel('Correlation');
legend(audioRepresentations, 'Location', 'Best');
title('Prediction Performance with Different Audio Representations (FindTRF Shrinkage)');
%%
save TestDemoData kValues regular_results methodValues testmodel_dnn ...
    self_correlation meanmodel_correlation ...
    cleanmodel_correlation random_correlation
