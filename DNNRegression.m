function [g,pred] = DNNRegression(stimulus, response, direction, ...
    testdata, g, lags, valid, verbosity)
% function [g,pred] = DNNRegression(stimulus, response, direction, ...
%     testdata, g, lags, valid, verbosity)
if direction > 0
    dirArg = 'forward';
else
    dirArg = 'reverse';
end
if isempty(g) || strcmp(g, '')
    modelDataFile = [tempname() '.pkl'];
else
    modelDataFile = g;
end
if ~exist('lags', 'var'), lags = 25; end;
if ~exist('valid', 'var'), valid = []; end;
if ~exist('verbosity','var'), verbosity = 0; end;


cmdName = which('DNNregression.py');    % Look for the python script
if isempty(cmdName), error('Can''t find the DNNregression.py file.'); end;
yamlName = which('network.yaml');        % Look for network description
if isempty(yamlName), error('Can''t find the network.yaml file.'); end;

% If we have it, save the valid file.
% QUESTION?  Is it only valid during training?
if ~isempty(valid)
    validDataFile = [tempname() '.mat'];
    data = valid;
    save(validDataFile, 'data');
else
    validDataFile = [];
end

if ~isempty(stimulus) && ~isempty(response)
    % Model prediction function calls looks like this:
    %   python DNNregression.py -t -m "./network.yaml"
    %       -s "trainUnattendedAudio.mat" \
    %       -r "trainResponse.mat" -w "./network_best_3.pkl" --debug
    
    if ischar(stimulus)
        stimulusDataFile = stimulus;
    else
        stimulusDataFile = [tempname() '.mat'];
        data = stimulus;
        save(stimulusDataFile, 'data');
    end
    if ischar(response)
        responseDataFile = response;
    else
        responseDataFile = [tempname() '.mat'];
        data = response;
        save(responseDataFile, 'data');
    end
    clear data;
    cmd = sprintf('python "%s" -t --dir %s -m "%s" -s "%s" -r "%s" -w "%s" --context %d --verbosity %d', ...
        cmdName, dirArg, yamlName, stimulusDataFile, responseDataFile, ...
        modelDataFile, lags, 0);
    if ~isempty(validDataFile)
        cmd = sprintf('%s --valid "%s"', cmd, validDataFile);
    end
    if verbosity
        fprintf('Executing the training command: %s"\n', cmd);
        system(cmd);
    else
        [err,status] = system(cmd);
        if err
            % If you get to this point, you can see the PyLearn2 output
            % in the variable status.
            error('DNN training command failed.  Rerun with verbosity=1 to see the error.');
        end
    end
end

if ~isempty(testdata)
    if ischar(testdata)
        testDataFile = testdata;
    else
        testDataFile = [tempname() '.mat'];
        data = testdata;
        save(testDataFile, 'data');
        clear data;
    end
    predictionDataFile = [tempname() '.mat'];
    if direction >= 0             % Given stimulus, predict response
        inputFlag = '-s';
        outputFlag = '-r';
    else                    % Given response, predict stimulus
        inputFlag = '-r';
        outputFlag = '-s';
    end
    cmd = sprintf('python "%s" -p --dir %s -m "%s" %s "%s" %s "%s" -w "%s" --context %d --verbosity %d', ...
        cmdName, dirArg, yamlName, inputFlag, testDataFile, outputFlag, predictionDataFile, ...
        modelDataFile, lags, 0);
    if verbosity
        fprintf('Executing the prediction command: %s\n', cmd);
        system(cmd);
    else
        [err,status] = system(cmd);
        if err
            % If you get to this point, you can see the PyLearn2 output
            % in the variable status.
            error('DNN prediction command failed.  Rerun with verbosity=1 to see the error.');
        end
    end
    predictionResult = load(predictionDataFile);
    pred = predictionResult.data;
end

g = modelDataFile;

if 0
    %%
    lags = round(1.5*impulseLength*fs);
    dnnTrainingWindow = 2;  % Seconds on each side of the attention switch
    iTrain = find(recordingT > attentionDuration - dnnTrainingWindow & ...
        recordingT < attentionDuration + dnnTrainingWindow);
    iTest = find(recordingT > 2*attentionDuration & ...
        recordingT < recordingT(end-lags));
    dnnDirection = -1;
     % lags = 1;
    
    % Now calculate the models for the attended and unattended signals.
    verbosity = 1;
    attentionModel = DNNRegression(attendedAudio(iTrain), response(iTrain, :), ...
        dnnDirection, [], [], lags, verbosity);
    unattentionModel = DNNRegression(unattendedAudio(iTrain), response(iTrain, :), ...
        dnnDirection, [], [], lags, verbosity);
    
    %%
    [~, attendedPrediction] = DNNRegression([], [], ...
        dnnDirection, response, attentionModel, lags);
    [~, unattendedPrediction] = DNNRegression([], [], ...
        dnnDirection, response, unattentionModel, lags);
    
    ca = corrcoef([attendedAudio(iTest) attendedPrediction(iTest)]);
    cu = corrcoef([unattendedAudio(iTest) unattendedPrediction(iTest)]);
    fprintf('Attended correlation: %g, Unattended correlation: %g.\n', ...
        ca(1,2), cu(1,2));
    %%
    % Plot the predicted stimuli
    clf
    attentionSwitchPick = 3;
    iPlot = find(recordingT>attentionSwitchPick*attentionDuration-dnnTrainingWindow & ...
        recordingT < attentionSwitchPick*attentionDuration+dnnTrainingWindow);
    plot(recordingT(iPlot), [attendedPrediction(iPlot) unattendedPrediction(iPlot)]')
    legend('Attended Signal', 'Unattended Signal');
    title('Predicted Signals');
    xlabel('time (seconds)'); ylabel('Intensity');
    axis tight
    
    %%
    % Plot the matrix of measured/predicted and attended/unattended signals
    clf
    subplot(2, 2, 1);
    plot(recordingT(iPlot), attendedAudio(iPlot()));
    title('Attended Signal'); axis tight
    
    subplot(2, 2, 2);
    plot(recordingT(iPlot), unattendedAudio(iPlot()));
    title('Unattended Signal'); axis tight
    
    subplot(2, 2, 3);
    plot(recordingT(iPlot), attendedPrediction(iPlot()));
    title('Attended TRF Prediction'); axis tight
    
    subplot(2, 2, 4);
    plot(recordingT(iPlot), unattendedPrediction(iPlot()));
    title('Unattended TRF Prediction'); axis tight
    
end
