function newS = resample_fft(oldS, oldFs, newFs)
% function newS = resample_fft(oldS, oldFs, newFs)
% 
% Resample the columns of a 2D matrix, one row at a time. The input parameter
% oldS is the array of data, one signal per column, oldFs is the original
% (old) sample rate, newFs is the new (desired) sample rate.

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com


newS = [];
for col=1:size(oldS,2)
    newPiece = resample_fft_1d(oldS(:,col), oldFs, newFs);
    if isempty(newS)
        newS = zeros(size(newPiece,1), size(oldS,2));
    end
    newS(:,col) = newPiece;
end


function newS = resample_fft_1d(oldS, oldFs, newFs)
% function newS = resample_fft(oldS, oldFs, newFs)
%
% Resample an (old) signal to a new sample rate.  Original (old) sample
% rate is oldFs, new sample rate is newFs. This code only processes
% one-dimensional signals, 1xN samples in size.
%
% This code uses an FFT to perform the interpolation.  When the sample rate
% is reduced, reduce the size of the size of the FFT by cutting off the
% high frequency components.  When the sample rate is increased, zero pad
% the FFT to extend the signal.

if nargin < 1
    oldFs = 10;
    oldS = sin(0:1/oldFs:9.999)';
    newFs = 5;              % 5 and 20 are good values to try
    newS = resample_fft_1d(oldS, oldFs, newFs);
    oldT = (0:length(oldS)-1)'/oldFs;
    newT = (0:length(newS)-1)'/newFs;
    
    handles = plot(oldT, oldS, oldT, oldS, 'o', ...
        newT, newS, newT, newS, 'x');
    xlabel('Time ->');
    legend(handles([1 3]), 'Original', 'Resampled');
    return;
end
extendedS = [oldS; 0*oldS];         % zero pad the data.

extN = length(extendedS);
extN2 = extN/2;
extN2p = extN - extN2;

newN = round(extN*newFs/oldFs);
newN2 = floor(newN/2);
newN2p = newN2;  % newN - newN2;

fOld = fft(extendedS);

fNew = zeros(newN, 1);
if newFs > oldFs
    % Upsample: Need to add zeros in the middle
    fNew(1:extN2) = fOld(1:extN2);
    fNew(end-extN2p+1:end) = fOld(end-extN2p+1:end);
else
    % Downsample: Need to keep LF coefficients
    fNew(1:newN2) = fOld(1:newN2);
    fNew(end-newN2p+1:end) = fOld(end-newN2p+1:end);
end
newS = ifft(fNew)*newFs/oldFs;
newS = real(newS(1:newN2));
if sum(isnan(newS)) > 0
    error('Got a NaN in resample_fft_1d');
end


if 0
    %%
    % Create sum of two sinusoids (1/2pi Hz and 10/pi Hz) sampled at 100Hz
    t = (0:.01:4*pi)';
    s = sin(t) + sin(20*t);
    subplot(3,1,1); plot(t,s); axis tight; ylabel('Original');
    title('Demonstration of resample\_fft');
    
    % Resample the signal down to 1Hz.
    s2 = resample_fft(s, 100, 1);
    subplot(3,1,2); plot(t(1:100:end-99),s2, t(1:100:end-99),s2, 'x');
    axis tight; ylabel('Down Sampled');
    
    % Upsample the down-sampled signal back to the original 100Hz
    s3 = resample_fft(s2, 1, 100);
    subplot(3,1,3); plot(t(1:length(s3)), s3); axis tight
    ylabel('Up Sampled');
    %%
end


    
