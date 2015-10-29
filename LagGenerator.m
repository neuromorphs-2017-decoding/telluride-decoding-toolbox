function out = LagGenerator(in, Lags)
% function out = LagGenerator(R, Lags)
%
% Shift a temporal response to create multiple copies of the original data,
% shifted in time so each row of the output response has several temporally
% shifted versions of the original data. Zero pads the first part of the
% output for positive lags and zero pads the last part for negative lags

% Inputs:
% in :      - is time x channels or time x features
% Lags :    - the time delays to use in samples
%
% Output: 
% out :     - is the time x (chans*lags)
%
% For example: LagGenerator([(1:5)' (2:2:10)'],-1:1)
% Output: the center two channels contains the original data.  The left
% two columns are shifted forward in time, while the right two columns are
% shifted backward

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

dimshift = size(in,2);
out = zeros(size(in,1), dimshift*length(Lags));

idx = 1;
for lag = Lags
    t1 = circshift(in,lag);
    if lag < 0
        t1(end-abs(lag)+1:end,:) = 0;   % zero out end of time
    else
        t1(1:lag,:) = 0;                % Zero out first part of time.
    end
    out(:,idx:idx+dimshift-1) = t1(1:size(in,1),:);
    idx = idx + dimshift;               % Advance to next lag slice
end
