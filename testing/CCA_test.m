% Some CCA tests

% First make sure we have the toolbox code.
if exist('cca', 'file') ~= 2
    path(path, '..')
end
%%
% Generate the last figure in the CCA appendix, showing the reconstruction
% accuracy.
N = 100000;

r1 = randn(N,1);            % Underlying (latent) signal
r2 = randn(N,1);            % Underlying (latent) signal
n1 = randn(N,2);
n2 = randn(N,3);

% First latent component is mapped into axis aligned coordinates
x1 = r1 * [1 0];
y1 = r1 * [0 1 0] + 0*randn(N,3);

% Second latent component is mapped into a three-dimensional space.
x2 = r2 * [1 1];
y2 = r2 * [1 -1 0] + 0.0*randn(N,3);

% Create the summed signal, by combining the two sets of random processes
gain = .3;
s1 = x1 + gain*x2 + n1/10;
s2 = y1 + gain*y2 + n2/10;

method = 'matlab';
switch method
    case 'toolbox'
        fprintf('Using Toolbox CCA\n');
        [Wx, Wy, r] = cca(s1', s2');
        recon1 = s1 * Wx; % recon1 = recon1/std(recon1);
        recon2 = s2 * Wy; % recon2 = recon2/std(recon2);
    case 'ccny'
        fprintf('Using CCNY CCA\n');
        [A,B,rhos,pvals,U,V] = cca_ccny(s1', s2');
        recon1 = s1 * A;
        recon2 = s2 * B;
    case 'press'
        fprintf('Using CCA_press test\n');
        [a, b, s] = cca_press(s1, s2);

        recon1 = s1 * a; % recon1 = recon1/std(recon1(:));
        recon2 = s2 * b; % recon2 = recon2/std(recon2(:));
     case 'matlab'
        fprintf('Using MathWorks Canoncorr\n');
        [a, b, s] = canoncorr(s1, s2);

        recon1 = s1 * a; % recon1 = recon1/std(recon1(:));
        recon2 = s2 * b; % recon2 = recon2/std(recon2(:));
end

if 1
    % figure out whether the two signals got swapped.
    c11 = corrcoef([r1 recon1(:, 1)]); c11 = c11(1,2);
    c12 = corrcoef(r1, recon1(:, 2)); c12 = c12(1,2);
    c21 = corrcoef(r2, recon1(:, 1)); c21 = c21(1,2);
    c22 = corrcoef(r2, recon1(:, 2)); c22 = c22(1,2);
    c = [c11 c12; c21 c22]

    if abs(c11)  < abs(c12)
        fprintf('Need to flip components\n');
        t = r2;
        r2 = r1;
        r1 = t;
    end
end


% CCA will rotate the two datasets to match, and they will have the 
% strongest possible correlation.  But there is no guarantee that the CCA
% algorithm will choose the same sign and gain as the latent variable.  
% Compute the dot product and scale to "fix" the gain for better plotting.
recon1(:,1) = recon1(:,1) / (sum(recon1(:,1).*r1) / sum(r1.*r1));
recon2(:,1) = recon2(:,1) / (sum(recon2(:,1).*r1) / sum(r1.*r1));
recon1(:,2) = recon1(:,2) / (sum(recon1(:,2).*r2) / sum(r2.*r2));
recon2(:,2) = recon2(:,2) / (sum(recon2(:,2).*r2) / sum(r2.*r2));

% Plot the results.
np = 1:30;
subplot(2, 1, 1);
plot(np, r1(np), np, recon1(np, 1), 'x', np, recon2(np, 1), 'o')
title('Reconstruction of first CCA component compared to original')
legend('Original', 'Reconstruction from dataset 1', 'Reconstruction from dataset 2');
xlabel('Sample #')

subplot(2, 1, 2);
plot(np, r2(np), np, recon1(np, 2), 'x', np, recon2(np, 2), 'o')
title('Reconstruction of second CCA component compared to original')
legend('Original', 'Reconstruction from dataset 1', 'Reconstruction from dataset 2');
xlabel('Sample #')

c11 = corrcoef([r1 recon1(:, 1)]); c11 = c11(1,2);
c12 = corrcoef(r1, recon1(:, 2)); c12 = c12(1,2);
c21 = corrcoef(r2, recon1(:, 1)); c21 = c21(1,2);
c22 = corrcoef(r2, recon1(:, 2)); c22 = c22(1,2);


%%
% Compare the speed and accuracy of the different implementations.
N=10000;
D=1280;
REPS=25;
methods = {'toolbox', 'press', 'matlab', 'ccny'};
for method = methods
    method = method{1};
    tic;
    for i=1:REPS
        X = randn(N,D);
        Y = randn(N,D);
        switch method
            case 'toolbox'
                [Wx, Wy, r] = cca(s1', s2');
            case 'press'
                [a, b, s] = cca_press(s1, s2);
            case 'matlab'
                [a, b, s] = canoncorr(s1, s2);
            case 'ccny'
                [A,B,rhos,pvals,U,V] = cca_ccny(s1', s2');
        end
    end
    averageTime = toc/REPS;
    fprintf('Time for method %s is %gs.\n', method, averageTime);
end

%%
% Test the reconstruction accuracy vs. noise level
N = 100000;

r1 = randn(N,1);            % Underlying (latent) signal
r2 = randn(N,1);            % Underlying (latent) signal
n1 = randn(N,2);
n2 = randn(N,3);

% First latent component is mapped into axis aligned coordinates
x1 = r1 * [1 0];
y1 = r1 * [0 1 0] + 0*randn(N,3);

% Second latent component is mapped into a three-dimensional space.
x2 = r2 * [1 1];
y2 = r2 * [1 -1 0] + 0.0*randn(N,3);

gains = 2.^(1:-1:-6);
for gi = 1:length(gains)
    gain = gains(gi);
    % Create the summed signal, by combining the two sets of random processes
    s1 = x1 + gain*x2 + n1/10;
    s2 = y1 + gain*y2 + n2/10;

    method = 'toolbox';
    switch method
        case 'toolbox'
            fprintf('Using Toolbox CCA\n');
            [Wx, Wy, r] = cca(s1', s2');
            recon1 = s1 * Wx; % recon1 = recon1/std(recon1);
            recon2 = s2 * Wy; % recon2 = recon2/std(recon2);
        case 'ccny'
            fprintf('Using CCNY CCA\n');
            [A,B,rhos,pvals,U,V] = cca_ccny(s1', s2');
            recon1 = s1 * A;
            recon2 = s2 * B;
        case 'press'
            fprintf('Using CCA_press test\n');
            [a, b, s] = cca_press(s1, s2);

            recon1 = s1 * a; % recon1 = recon1/std(recon1(:));
            recon2 = s2 * b; % recon2 = recon2/std(recon2(:));
         case 'matlab'
            fprintf('Using MathWorks Canoncorr\n');
            [a, b, s] = canoncorr(s1, s2);

            recon1 = s1 * a; % recon1 = recon1/std(recon1(:));
            recon2 = s2 * b; % recon2 = recon2/std(recon2(:));
    end

    if 1
        % figure out whether the two signals got swapped.
        c11 = corrcoef([r1 recon1(:, 1)]); c11 = c11(1,2);
        c12 = corrcoef(r1, recon1(:, 2)); c12 = c12(1,2);
        c21 = corrcoef(r2, recon1(:, 1)); c21 = c21(1,2);
        c22 = corrcoef(r2, recon1(:, 2)); c22 = c22(1,2);
        c = [c11 c12; c21 c22];

        if abs(c11) < abs(c12)
            fprintf('Need to flip components\n');
            t = r2;
            r2 = r1;
            r1 = t;
        end
    end


    % CCA will rotate the two datasets to match, and they will have the 
    % strongest possible correlation.  But there is no guarantee that the CCA
    % algorithm will choose the same sign and gain as the latent variable.  
    % Compute the dot product and scale to "fix" the gain for better plotting.
    recon1(:,1) = recon1(:,1) / (sum(recon1(:,1).*r1) / sum(r1.*r1));
    recon2(:,1) = recon2(:,1) / (sum(recon2(:,1).*r1) / sum(r1.*r1));
    recon1(:,2) = recon1(:,2) / (sum(recon1(:,2).*r2) / sum(r2.*r2));
    recon2(:,2) = recon2(:,2) / (sum(recon2(:,2).*r2) / sum(r2.*r2));

    c11 = corrcoef([r1 recon1(:, 1)]); c11 = c11(1,2);
    c12 = corrcoef(r1, recon1(:, 2)); c12 = c12(1,2);
    c21 = corrcoef(r2, recon1(:, 1)); c21 = c21(1,2);
    c22 = corrcoef(r2, recon1(:, 2)); c22 = c22(1,2);
    performance(gi, :) = [c11 c12 c21 c22];
end

semilogx(gains, performance)
axis tight
xlabel('Relative gain between signal 1 and 2');
ylabel('Correlation between reconstruction and original signal')
title('Reconstruction Correlations with Hidden Signal')
legend(...
    'Correlation of result 1', 'Correlation between signals 1 and 2', ...
    'Correlation between signals 1 and 2', 'Correlation of result 1', ...
    'location' ,'east')