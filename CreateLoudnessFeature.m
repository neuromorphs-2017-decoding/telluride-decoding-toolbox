function loudnessFeature = CreateLoudnessFeature(audioData, audioFS, loudnessFS)
% function loudnessFeature = CreateLoudnessFeature(audioData, audioFS, loudnessFS)
% Compute the loudness (intensity) of an audio signal by averaging the
% squared energy.  
%
% Given audioData at a sample rate of audioFS, compute the loudnessFeature
% of the sound.  This is done by finding the RMS energy within +/-1
% windowSize (defined below) samples of the center point.  Then, if desired
% average this calculation over delta frames to smooth the energy
% calculation.  

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

audioEnergy = audioData .^ 2;
windowSize = 1.5/loudnessFS;          % in seconds

N = round(length(audioData)/audioFS*loudnessFS);
loudnessFeature = zeros(N, 1);

for i=1:N
    t = i/loudnessFS;                 % Center of window In seconds
    b = max(1,round((audioFS*(t-windowSize))));
    e = min(length(audioData), ...
            max(1, round((audioFS*(t+windowSize)))));
    m = mean(audioEnergy(b:e)); 
    if isnan(m)
        error('Got NaN when computing Loudness Feature');
    end
    loudnessFeature(i) = m;
end

if 0            % Test Code
    [tap,fs] = audioread('tapestry.wav');
    lfs = 100;
    loudness = CreateLoudnessFeature(tap', fs, lfs);
    h  = plot((1:length(tap))/fs, tap, ...
        (1:length(loudness))/lfs, sqrt(loudness));
    xlabel('Seconds'); ylabel('Amplitude');
    
    set(h(2), 'LineWidth', 6)
end
