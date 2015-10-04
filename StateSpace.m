function [q , q_L, q_U ] = StateSpace2(corr_m, corr_f, max_iterations, debug)
% function [q , q_L, q_U ] = StateSpace_New(g_a, g_u, TestStim, TestEEG , Fs, TRF_Method, Dir, K, doScale, Lags)

%This function estimates the probability of attending to attending to spk1
%in a two-speaker environment.

%OUTPUTS:
%q : Probability of attending to spk1
%q_L: Lower bound for %70 confidence interval
%q_U: Upper bound for %70 confidence interval

%INPUTS:
%g:(lags x features x channels) : the TRF If empty, it will be calculated from training data.
%Input a pretrained g to predict based on testdata (specify same lags as used in training).
%If stim/resp is provided then g is always estimated Singleton dimensions are removed

%Lags:Are the time delays between stimulus and response in samples specified [start end] or a full vector of lags
%Should be positive for forward and negative for backward mapping. Default is 0 to 100

%dir =  1 : forward mapping (predict neural response from stimulus) (encoding)
%dir = -1 : backward mapping (predict stimulus from neural response) (decoding)

%TestStim : Envelopes of the attended and unattended speakers (2*time)

%TestEEG : test EEG (time*channels*trials)

%Fs: Sampling Frequency (Hz)
%
% Sahar Akram, sahar.akram@gmail.com
% (Akram et. al., A State-Space Model for Decoding Auditory Attentional Modulation from MEG
%   in a Competing-Speaker Environment, NIPS 2014)

if nargin < 3
    max_iterations = 300; % Total number of EM iterations
end
if nargin < 4
    debug = 0;
end

%EM params
epochSize = 50; % Analysis time window
lambda = 1; % state transition scale
kappa1 = .5; % von-Mises param for spk1
kappa2 = .5; % von-Mises param for spk2
kreg = 0.01; %Regularization param
beta = .2; % inverse Gamma Scale
alpha = 2.01; % inverse Gamma shape


%% Computing correlation values
% 
T = size(corr_m, 1);
nEpochs = size(corr_m, 1);
% nCH = size(TestEEG,2);
nT= size(corr_m, 2);
% 
% 
% corr_m(1:nEpochs,1:nT) = 0;
% corr_f(1:nEpochs,1:nT) = 0;
% for j = 1:nT
%     for l=1:nEpochs
%         [~, attendedPrediction] = FindTRF([], [],Dir, squeeze(TestEEG((l-1)*epochSize+1:l*epochSize,:,j)), g_a, Lags, TRF_Method, K, doScale);
%         [~, unattendedPrediction] = FindTRF([], [],Dir, squeeze(TestEEG((l-1)*epochSize+1:l*epochSize,:,j)), g_u, Lags, TRF_Method, K, doScale);
%         corr_m(l,j) = corr(TestStim(1,(l-1)*epochSize+1:l*epochSize)', attendedPrediction);
%         corr_f(l,j) = corr(TestStim(2,(l-1)*epochSize+1:l*epochSize)', unattendedPrediction);
%         
%     end;
% end

corr_m(isnan(corr_m))=0;
corr_f(isnan(corr_f))=0;
% The EM algorithm

g_s(1:nEpochs) = 0;
t(1:nEpochs, 1:nT) = 0;

for m=1:max_iterations
    for j=1:nT
        for l=1:nEpochs
            pp = exp(g_s(l))/(1+exp(g_s(l)));
            t1 = corr_m(l,j);
            t2 = corr_f(l,j);
            t(l,j) = pp*(kappa1^(epochSize/2-1))*exp(kappa1*t1)/besseli(epochSize/2-1,kappa1) /(pp*(kappa1^(epochSize/2-1))*exp(kappa1*t1)/besseli(epochSize/2-1,kappa1) + (1-pp)*(kappa2^(epochSize/2-1))*exp(kappa2*t2)/besseli(epochSize/2-1,kappa2));
        end;
    end
    
    %%
    %Plot the output of each EM iteration (Please uncomment if you want to take a look at the output after each iteration step)
    
    if debug
        plot(exp(g_s)./(1+exp(g_s)),'b'); axis([1 length(g_s) 0 1]);
        xlabel('Epoch Number'); ylabel('Probability of Speaker 1');
        title(sprintf('Progress in State-Space Decoding, Iteration %d', m));
        drawnow
        % pause
    end
    %%
    
    % updating von Mises scales
    
    k1 = 0;
    k2 = 0;
    
    for j = 1:nT
        for l=1:nEpochs
            t1 = corr_m(l,j);
            t2 = corr_f(l,j);
            k1 = k1 + t(l,j)*t1;
            k2 = k2 + (1-t(l,j))*t2;
        end
    end;
    
    d=100*size(t,1)*size(t,2)/2;
    denom1=d+sum(sum(t));
    denom2=d+sum(sum(1-t));
    
    k0 = kappa1;
    kappa11=kappa1;
    opts = optimset('Display','off');
    kappa1 = fsolve(@(kappa11)vMF(kappa11, epochSize/2,(1/denom1)*(k1+d*kreg)), k0, opts);
    
    k0 = kappa2;
    kappa22= kappa2;
    opts = optimset('Display','off');
    kappa2 = fsolve(@(kappa22)vMF(kappa22, epochSize/2,(1/denom2)*(k2+d*kreg)), k0, opts);
    
    
    %Run non-filter and smoother
    sigma_pp = ones(nEpochs,1)*2*beta/(1+2*(alpha+1)); % state hyperparameter for each time block
    
    for h=1:5
        
        g_km(1:nEpochs) = 0;
        g_kk(1:nEpochs) = 0;
        sp_km(1:nEpochs) = .1;
        sp_kk(1:nEpochs) = .1;
        
        for k=2:nEpochs
            g_km(k) = lambda*g_kk(k-1);
            sp_km(k) = lambda^2*sp_kk(k-1) + sigma_pp(k)/nT;
            g_kk(k) = g_km(k) + sp_km(k)*(sum(t(k,:)') - nT*exp(g_km(k))/(1+exp(g_km(k))));
            
            for newt_iter=1:10
                g_kk(k) = g_kk(k) - (g_kk(k) - g_km(k) - sp_km(k)*(sum(t(k,:)') - nT*exp(g_kk(k))/(1+exp(g_kk(k)))))/(1 + sp_km(k)*nT*exp(g_kk(k))/((1+exp(g_kk(k)))^2));
                g_kk(k);
            end;
            
            sp_kk(k) = 1/(1/sp_km(k) + nT*exp(g_kk(k))/(1+exp(g_kk(k)))^2);
        end;
        
        % PP smoother
        
        g_s(1:nEpochs) = 0;
        sp_s(1:nEpochs) = 0;
        g_s(end) = g_kk(end);
        sp_s(end) = sp_kk(end);
        
        for k=nEpochs-1:-1:1
            ss = sp_kk(k)*lambda/sp_km(k+1);
            g_s(k) = g_kk(k) + ss*(g_s(k+1) - g_km(k+1));
            sp_s(k) = sp_kk(k) + ss^2*(sp_s(k+1) - sp_km(k+1));
        end;
        
        
        % update the state covariance
        
        for l=2:nEpochs
            sigma_pp(l) = (2*beta + ((g_s(l) - lambda*g_s(l-1))^2 + sp_s(l) + lambda^2*sp_s(l-1) - 2*lambda*sp_s(l)*sp_kk(l-1)*lambda/sp_km(l)))/(1+2*(alpha+1));
        end;
        
        sigma_pp(1) = (2*beta + ((g_s(1))^2 + sp_s(1)))/(1+2*(alpha+1));
    end
end

%Assign outputs
q = exp(g_s)./(1+exp(g_s));
q_U = exp(g_s +1.04*sqrt(sp_s))./(1+exp(g_s +1.04*sqrt(sp_s)));
q_L = exp(g_s -1.04*sqrt(sp_s))./(1+exp(g_s -1.04*sqrt(sp_s)));



