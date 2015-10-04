function [g,pred] = FindTRF(stimulus, response, Dir, testdata, g, Lags, Method, K, doscale)
% FindTRF - Find the temporal response function connecting stimulus and
% response.
% [g,pred] = TRF(stimulus, response, Dir, testdata, g, Lags, Method, K, doscale)
% 
%   Find Temporal Response Function (FindTRF). Performs regression to find g that maps
%   linearly between stimulus and response. The TRF is a function of time lag 
%   between stimulus and response. Dir specifies the direction of the mapping. 
%   Dir =  1 -> forward mapping (predict neural response from stimulus) (encoding) 
%   Dir = -1 -> backward mapping (predict stimulus from neural response) (decoding)
%
%   Forward mapping uses a stimulus in testdata to predict a neural response.
%   Backward mapping uses a response in testdata to reconstruct a stimulus. 
%   Method is the regularization method used to prevent overfitting and K is a 
%   regularization parameter.
% 
%   Inputs:
%      stimulus - Stimulus training set (time x 1 or time x features)
%      response - Neural responses  (time x 1 or time x channels)
% 
%   Optional:
%      Dir      - Direction 1 (forward mapping) or -1 (backward mapping)
%      testdata - (time x features) : test data used to predict
%      g        - (lags x features x channels) : the TRF 
%                    If empty, it will be calculated from training data. 
%                    Input a pretrained g to predict based on testdata
%                    (specify same lags as used in training).
%                    If stim/resp is provided then g is always estimated
%                    Singleton dimensions are removed
%      Lags     - Are the time delays between stimulus and response in samples
%                    Specify [start end] or a full vector of lags 
%                    Lags are always positive for causal data 
%                    (ie. the stimulus comes before the response)
%                    Default is 0 to 100
%      Method   - Regularization method to use. Use this to remove noise and 
%                    avoid overfitting
%                 An additional input K specifies the regularization term
%                 Possible inputs:
%                 - 'Shrinkage', K (Default) (default K=0.2)
%                 - 'Ridge', K (default K=10) : ridge regression with penalty
%                      on neighbouring lags (see Lalor et al 2006: The VESPA) 
%                 - 'NRC', K (default K=0.99) :  normalized reverse correlation
%                 - 'Lasso', K (default K=[.01 1]) : lasso / elastic net.
%                      Uses matlab's lasso (needs statistics toolbox).
%                      K(2) controls the degree of L1 penalty (elastic net)
%                      This option may result in long computation times
%                 - 'None' : no regularization, use ordinary least-squares
%      K          - Regularization parameter(s)
%      doscale    - Option to scale inputs to zero mean and std 1 (Default on)
% 
%   Returns:
%      g: the TRF function.  It will have a size of
%           nlags, nfeatures, nchannels
%      pred: predicted output from testdata
% 
% hac @ telluride2015
% version 09-24-2015 

if exist('Method', 'var') 
    Method = lower(Method);
else
    Method = 'shrinkage';
end

if ~exist('Lags','var') || isempty(Lags)
    Lags = 0:100;
elseif length(Lags)==2
    Lags = Lags(1):Lags(2);
end
nlag = length(Lags);
if ~exist('Dir','var') || isempty(Dir) || Dir > 0 
    Dir = 1;
elseif Dir < 0
    Lags = -1*fliplr(Lags);
end

if ~exist('testdata','var') || isempty(testdata)
    testdata=[];
    pred = [];
end

if ~isempty(stimulus)
    if strcmp(Method,'lasso') && ~license('test', 'statistics_toolbox')
                error('Statistics toolbox needed for lasso')
        error('You need to have the statistics toolbox to use the Lasso method.');
    end
    
    % Figure out the inputs and outputs from the regression.  We are given
    % x and we want to find Y.
    if Dir > 0              % Forward direction, from stimulus to response
        x = stimulus;
        Y = response;
    elseif Dir < 0          % Backward, from response to stimulus
        x = response;
        Y = stimulus;
    else
        disp('Specify Dir 1 (forward) or -1 (backward)');
    end
    if ~isempty(testdata) && ~exist('g','var')
        if size(x,2) ~= size(testdata,2)
            error('Testdata does not have the same # of channels as stimulus');
        end
    end
    if ~exist('K', 'var') || isempty(K)
        switch Method
            case 'shrinkage'
                K=.2;
            case 'ridge'
                K=10;
            case 'nrc'
                K=.99;
            case 'lasso'
                K = [0.01,1];
        end
    end
    nx = size(x,2);
    ny = size(Y,2);
end

if ~exist('doscale', 'var')
    doscale = 1;
end

if ~exist('g','var') || isempty(g)
    g=[];
else
    if ~isempty(stimulus)
        fprintf('You have provided both g and stim/resp data. \nWe estimate a new g based on the training data\n')
    else
       if nlag ~= size(g,1)
           error('Number of Lags and g does not correspond')
       end

        if ~isempty(testdata) % reshape g for prediction
            ndim = ndims(g); 
            if Dir == 1 && ndim==3
                g = permute(g,[1 3 2]); % adds singleton for 2d   
            end
            nx = size(testdata,2);
            ny = size(g,ndim-1);
            if ndim==3
                g = shiftdim(g,ndims(g)-1);
                g = reshape(g,nlag*nx,ny);
            else
                if nx~=1
                    g = reshape(g',nlag*size(g,2),1);
                else
                    g = reshape(g,nlag,size(g,2));
                end
            end
        else
            error('Provide either training data or testdata')
        end
    end
end

if ~isempty(stimulus)
    
    x(isnan(x))=0;

    if doscale
        x = dozscore(x); 
        Y = dozscore(Y);
    end
        
    X = LagGenerator(x,Lags);
    
    XX = X' * X;
    XY = Y' * X;  

    switch Method
        case 'shrinkage'
            XX = (1-K)*XX + K*mean(eig(XX))*eye(length(XX));
            g=XX\XY';
        case 'ridge'
            M = 2*eye(size(XX))-diag(ones(size(XX,1)-1,1),1)-diag(ones(size(XX,1)-1,1),-1); M([1 end]) = 1;
            g=(XX+K*M)\XY';
        case 'nrc' 
            [u,s,v] = svd(XX);
            energy = cumsum(diag(s)./sum(diag(s)));
            limit = find(energy>K, 1);
            newDiag = 1./diag(s);
            if ~isempty(limit) && limit+1 <= length(newDiag)
                newDiag(limit+1:end) = 0;
            end
            RRinv = v*diag(newDiag)*u';
            g = RRinv * XY';            
        case 'lasso'
            if length(K)==1,K(2)=1;end
            for ii=1:size(XY,1)
                g(:,ii) = lasso(XX,XY(ii,:),'Lambda',K(1),'Alpha',K(2),'RelTol',1e-3);
            end
        case 'none'
            g=XX\XY';
        otherwise
            error('unknown method')
    end
end

if ~isempty(testdata)
    testdata(isnan(testdata))=0;
    if doscale
        testdata = dozscore(testdata);
    end
    testlag = LagGenerator(testdata,Lags);
    pred = testlag * g; 
end

% reshape to lags x features x channels, removing singletons
if ~isempty(stimulus)
    g = squeeze(reshape(g',ny,nx,nlag));g=shiftdim(g,ndims(g)-1);
    if ndims(g) == 3 && Dir == 1
        g = permute(g,[1 3 2]); 
    end
    if nx==1 && ny==1
        g=g';
    end
end


function out = LagGenerator(in, Lags)
    dimshift = size(in,2);
    out = zeros(size(in,1), dimshift*length(Lags));
    idx = 1;
    for lag = Lags
        t1 = circshift(in,lag);
        if lag < 0
            t1(end-abs(lag)+1:end,:) = 0;   % zero out end of time
        else
            t1(1:lag,:) = 0;                % zero out first part of time.
        end
        out(:,idx:idx+dimshift-1) = t1(1:size(in,1),:);
        idx = idx + dimshift;               % Advance to next lag slice
    end
end

function out = dozscore(in)
        muin = mean(in);
        stdin = std(in,0,1);
        stdin = max(stdin, max(stdin)/1000);    
        in = bsxfun(@minus,in,muin);
        out = bsxfun(@rdivide,in,stdin);
end

% Examples:
if 0  % example using forward mapping:
    %%
    N = 1000;
    TrainStim=randn(N,1); % simulate training stimulus
    lags = 1:100; noiselevel = 1.4; origlags = [50 64];
    TrainStimLag = LagGenerator(TrainStim,lags); % lagged version of stimulus
    clear TrainResp
    TrainResp(:,1) = TrainStimLag(:,origlags(1)) + noiselevel*randn(N,1); % simulated response at lag 1
    TrainResp(:,2) = TrainStimLag(:,origlags(2)) + noiselevel*randn(N,1); % simulated response at lag 2
    
    [g] = FindTRF(TrainStim, TrainResp, 1, TrainStim, [], lags, 'ridge',[],1);
    
    clf; subplot(2,1,1);
    plot(lags,g)
    xlabel('Delta Time');
    ylabel('Stimulus Response');
    disp('You should have peaks in the impulse responses at samples 50 and 64');

    testStim = TrainStim + noiselevel*randn(size(TrainStim)); % test stimulus with new noise
    [~,predResp] = TRF([], [], 1, testStim, g, lags, 'NRC',[],1);
    
    subplot(2,1,2);
    t=501:650; plot(t, TrainResp(t,1)/max(TrainResp(:)), t, predResp(t)/max(predResp(:)),'r')
    legend('Original response', 'Predicted response');
    cc=corrcoef(TrainResp(:),predResp(:));
    fprintf('Correlation between original and predicted response: %1.2f\n',cc(2))
end

if 0 % Malcolm's deterministic stimulus reconstruction example
    %%
    N = 10000;
    StimulusData = randn(N,2);      % Generate a bunch of 2-channel data
    lags = 0:10; Dir = -1;
    
    % Create the response.  First channel is 0.5 times current time, plus
    % 0.25 times previous time.  Second channel is all noise.
    ResponseData = [StimulusData(:,1)*.5 + [StimulusData(2:end,1)*.25; 0] randn(N,1)];
    TestingTimes = 1:1000;          % Use first part for testing
    TrainingTimes = max(TestingTimes)+1:N;  % Use rest of data for training
    TrainingStim = StimulusData(TrainingTimes,:);
    TrainingResp = ResponseData(TrainingTimes,:);
    TestingStim = StimulusData(TestingTimes, :);
    TestingResp = ResponseData(TestingTimes, :);
    
    % Now compute the response.
    K = 1.0; doScale = 0;
    [g,TestingStimPredict] = FindTRF(TrainingStim, TrainingResp, Dir, ...
        TestingResp, [], lags, 'none', K, doScale);

    % Impulse response for feature 1, channel 1 should be powers of 2.
    subplot(2,1,1);
    plot(lags, g(:,1,1));
    xlabel('Delta Time');
    ylabel('Impulse Response');
    subplot(2,1,2);
    t=1:100; plot(t, TestingStim(t,1), t, TestingStimPredict(t),'x')
    legend('Ground Truth', 'Predictions');
end


end
