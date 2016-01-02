function [decoded_states,all_probs] = ViterbiSearch(observations, ...
	start_p, trans_p, emit_f)
% Viterbi Decoder, based on the algorithm described at:
%  https://en.wikipedia.org/wiki/Viterbi_algorithm
%
% observations has size N_times x D_obs
%   The observation vector has D_obs dimensions.
% start_p has size 1xN_states
% trans_p has size N_states x N_states (from, to)
% emit_f(obs_vec, s) is a function that take observations 
%   (obs_vec, a 1 x D_obs vector) and returns the probability for the
%   desired state (s).
% 
% ToDo: Need to convert entire calculation to logs.
%
% Malcolm Slaney, Google Machine Hearing Group, November 2015
% malcolm@ieee.org malcolmslaney@google.com

n_states = length(start_p);
n_times = size(observations, 1);

all_probs = zeros(n_times, n_states);   % Store the accumlated probabilities.
best_path = zeros(n_times, n_states);   % Store the best states

for y=1:n_states                        % First time step
    all_probs(1, y) = start_p(y) * emit_f(observations(1, :), y);
    best_path(1, y) = y;
end
    
for t = 2:n_times       % For all the remaining time steps
    for y = 1:n_states  %   For each state
        obs_p = emit_f(observations(t, :), y);
        % disp(all_probs(t-1,:)); disp(trans_p(:,y)'); disp(obs_p);
        probs = all_probs(t-1, :) .* trans_p(:,y)' .* obs_p;
        [prob, m] = max(probs);
        all_probs(t,y) = prob;
        best_path(t, y) = m;
    end
end

[~, decoded_states(size(observations,1), 1)] = max(all_probs(end,:));
for t=n_times-1:-1:1
    decoded_states(t) = best_path(t, decoded_states(t+1));
end

if 0
    %%
    % Test example from: https://en.wikipedia.org/wiki/Viterbi_algorithm
    % states = ('Healthy', 'Fever')
    % observations = ('normal', 'cold', 'dizzy')
    observations = [1 2 3]';
    start_p = [0.6 0.4];
    trans_p = [0.7 0.3; 0.4 0.6];


    emit_p = [0.5 0.4 0.1; 0.1 0.3 0.6];
    emit_f = @(o, y) emit_p(y,o)';
    [decoded_states, all_probs] = ...
        ViterbiSearch(observations, start_p, trans_p, emit_f)
    % Correct answers are (from wikipedia example):
    %   decoded_states = [1 1 2]'
    %   all_probs = [0.3000    0.0400
    %                0.0840    0.0270
    %                0.0059    0.0151];

end
if 0
    %% 
    % Continuous observation example. Based on correlation signals from 
    % an attention-switching model.
    max_t = 100;
    noise_std = 0.5;            % How much noise to add to correlations
    model_std = 0.5;            % How much noise the model expects
    T = (1:max_t)';             % A time scale, 1Hz sampling rate
    start_p = [0.5 0.5];        % We don't know who is speaking first
    trans_time = 25;            % 1/trans_time is good approximation for 
				% transition probability
    trans_p = [1-1/trans_time 1/trans_time; 1/trans_time 1-1/trans_time];
    true_state = mod(floor(T/25),2)+1;
    correlations = [randn(max_t,1)*noise_std + (2-true_state) ...
			randn(max_t,1)*noise_std + (true_state-1)];
    correlations = min(1, max(-1, correlations));
    subplot(2,1,1); plot(correlations); title('Synthetic Correlations'); 
    legend('Speaker 1', 'Speaker 2', 'Location', 'best'); ylabel('Correlation');
    
    % Probability of being in a particular state is based on the 
    % correlation of observations. Probability of state i is a Gaussian of
    % width model_std, centered at 1. Thus the emission probability for 
    % state i is a function of element i of the observation (correlation) 
    % vector. 
    emit_f = @(obs_v, desired_state) ...
	exp(-((obs_v(desired_state)-1).^2/model_std.^2));
    [decoded_states, all_probs] = ...
        ViterbiSearch(correlations, start_p, trans_p, emit_f);
    subplot(2,1,2); hands = plot(T, decoded_states, '*', T, true_state, '--'); 
    set(hands(1), 'LineWidth', 2)
    title('Decoded State'); xlabel('Time');
end
