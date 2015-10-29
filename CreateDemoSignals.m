%%
% Script to create the test data we use to illustrate the algorithms in the
% Telluride Decoding Toolbox.

% Note: time is always the first dimension (since every EEG signal has
% time, but an unknown number of channels.)

% By Malcolm Slaney, Google Machine Hearing Project
% malcolm@ieee.org, malcolmslaney@google.com

% Set up some basic experimental parameters.
nEEGChannels = 32;      % Number of EEG channels in the response.
fs = 100;               % Audio and EEG sample rate in Hz
T = 100;                % Length of the experiment in seconds
useSinusoids = 1;

%%
% Create the attention signal, switching every attentionDuration seconds
signalLength = 100;     % Seconds
recordingT = (1/fs:1/fs:signalLength)';
attentionDuration = 25; % Time listening to each signal
attentionSignal = mod(floor(recordingT/attentionDuration),2);

%%

% Create the basic impulse response for each channel and give it a 
% realistic envelope
impulseLength = .25;    % Length of the TRF
impulseTimes = (0:1/fs:impulseLength)';
envelope = 30*impulseTimes.*exp(-impulseTimes*30);

attendedImpulseResponse = randn(length(impulseTimes), nEEGChannels) .* ...
    repmat(envelope, 1, nEEGChannels);
unattendedImpulseResponse = randn(length(impulseTimes), nEEGChannels) .* ...
    repmat(envelope, 1, nEEGChannels);

% Cut the magnitude of the unattended inpulse response so that noise will 
% have a bigger effect.
unattendedImpulseResponse = unattendedImpulseResponse/4;

%%
clf;
subplot(2,1,1);
image(impulseTimes, 1:nEEGChannels, attendedImpulseResponse'*32+32);
title('Attended Impulse Response');
xlabel('Time (s)');
ylabel('EEG Channel');

subplot(2,1,2)
image(impulseTimes, 1:nEEGChannels, unattendedImpulseResponse'*32+32)
title('Unattended Impulse Response');
xlabel('Time (s)');
ylabel('EEG Channel');

%%
% Create the two audio signals that the subject is listening to. Use 
% sinusoids if we are debugging, 
if useSinusoids
    audioS1 = sin(recordingT*2*pi*5);
    audioS2 = sin(recordingT*2*pi*7);
    audioHF = [audioS1 audioS2]; % Two signals to listen to
else
    audioLF = randn(ceil(length(recordingT)/10), 2);
    audioHF = fft_resample(audioLF, fs/10, fs);
    audioHF = audioHF(1:length(recordingT), :);
end

%%
% Create the signals that correspond to the attended and unattended audio,
% under control of the deterministic attentionSignal created above.
attentionMatrix = [1-attentionSignal attentionSignal];
attendedAudio = sum(attentionMatrix .* audioHF, 2);
unattendedAudio = sum((1-attentionMatrix) .* audioHF, 2);
if useSinusoids
    clf
    imagesc(recordingT, [1 2], [attendedAudio unattendedAudio]')
    plot(recordingT, ...
         [attendedAudio+1 unattendedAudio-1.1 2*attentionSignal+2.1])
    xlim([attentionDuration-1 attentionDuration+1]);
    xlabel('Time (seconds)');
    ylabel('Unattended Signal        Attended Signal            Attention Signal')
    title('Test Signals'); ylim([-2.2 4.3]); set(gca, 'YTick', []);
end
%%
% Now convolve the attended and unattended audio with the two different
% impulse responses to create the simulated EEG signals.
response = zeros( ...
    length(attendedAudio) + size(attendedImpulseResponse,1) - 1,....
    nEEGChannels);
for c=1:nEEGChannels
    attendedResponse = conv(attendedAudio, attendedImpulseResponse(:,c));
    unattendedResponse = conv(unattendedAudio, unattendedImpulseResponse(:,c));
    % Sum up the attended and unattended response, and then add noise.
    response(:,c) = attendedResponse + unattendedResponse + ...
        3*randn(size(attendedResponse));
end
