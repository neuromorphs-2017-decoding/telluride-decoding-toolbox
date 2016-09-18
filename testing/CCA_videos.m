% Generate the videos (GIF animations) used to describe a simple test
% set for CCA.

% http://www.mathworks.com/matlabcentral/answers/94495-how-can-i-create-animated-gif-images-in-matlab
N = 300;
x = 2*filter([.1 0 0], [1 -.9 0], randn(1,N)); % .^ (1/10);
x = 0.5 + 0.5*x/max(abs(x));
plot(x);
% Fix the size of the plot (and the gif)
%%
p = get(gcf, 'Position');
set(gcf,'Position', [p(1) p(2) 250 250])
%%
filename = 'CCA_Latent1.gif';

for i = 1:40
    clf
    vec_x = [0 0 0] * x(i);
    vec_y = [0 1 0] * x(i);
    h = mArrow3(vec_x,vec_y,'color','green','stemWidth',0.02,'facealpha',0.5);
    axis([-.5 .5 0 1]);
    axis off
    title('Latent Variable #1');
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', 0);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0);
    end
end

%%
filename = 'CCA_Latent2.gif';
noiseLevel = 0.02;
for i = 1:40
    clf
    vec_0 = [0 0 0] * x(i);
    vec_y0 = [0 1 0] * x(i) + noiseLevel*randn(1, 3);;
    vec_y1 = [1 0 0] * x(i) + noiseLevel*randn(1, 3);;
    h1 = mArrow3(vec_0,vec_y0,'color','red','stemWidth',0.015,'facealpha',0.5);
    h2 = mArrow3(vec_0,vec_y1,'color','blue','stemWidth',0.01,'facealpha',0.5);
    % line([0 vec_y0(1)], [0 vec_y0(2)], 'color', 'red');
    % line([0 vec_y1(1)], [0 vec_y1(2)], 'color', 'blue');
    axis([-1 1 -1 1]);
    axis off
    hold off
    title('Vector Variables 1 and 2');
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', 0);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', 0);
    end
end

%%
filename = 'CCA_Latent3.gif';
RotationCount=40;
blueAngle0 = 0;
redAngle0 = pi/2;
FinalBlueAngle = 5*pi-pi/2; blueAngle = blueAngle0;
FinalRedAngle = -5*pi-pi/2; redAngle = redAngle0;
rotationStart = RotationCount;
for i = 1:3*RotationCount
    if i <= rotationStart
        theTitle = 'Initial Vectors';
    elseif i > rotationStart && i < (rotationStart+RotationCount+1)
        % Update the angle if we are rotating
        % fprintf('%d %d\n', i, i > 10 && i < (10+RotationCount+1));
        blueAngle = (i-rotationStart)/RotationCount*(FinalBlueAngle-blueAngle0)+blueAngle0;
        redAngle = (i-rotationStart)/RotationCount*(FinalRedAngle-redAngle0)+redAngle0;
        theTitle = 'CCA Looking for Optimum Rotation';
    else
        theTitle = 'Rotated Vectors after CCA';
    end
    clf
    vec_0 = [0 0 0] * x(i);
    noiseLevel = 0.02;
    vec_y0 = [sin(blueAngle) cos(blueAngle) 0] * x(i) + noiseLevel*randn(1, 3);
    vec_y1 = [sin(redAngle) cos(redAngle) 0] * x(i) + noiseLevel*randn(1, 3);
    h1 = mArrow3(vec_0,vec_y0,'color','red','stemWidth',0.015,'facealpha',0.5);
    h2 = mArrow3(vec_0,vec_y1,'color','blue','stemWidth',0.01,'facealpha',0.5);
    % line([0 vec_y0(1)], [0 vec_y0(2)], 'color', 'red');
    % line([0 vec_y1(1)], [0 vec_y1(2)], 'color', 'blue');
    axis([-1 1 -1 1]);
    axis off
    hold off
    title(theTitle);
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount', inf, 'DelayTime', 0);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0);
    end
end

%%
x2 = 2*filter([.1 0 0], [1 -.9 0], randn(1,N)); % .^ (1/10);
x2 = 0.3 + 0.3*x/max(abs(x));

filename = 'CCA_Latent4.gif';
noiseLevel = 0.02;

for i = 1:40
    clf
    noiseLevel = 0.02;
    vec_x1 = [0 0 0];
    vec_x2 = [0.1 0 0];
    vec_y1 = [0.0 x(i) 0] + noiseLevel*randn(1, 3);
    vec_y2 = [0.1 x(i) 0] + noiseLevel*randn(1, 3);
    h = mArrow3(vec_x1,vec_y1,'color','red','stemWidth',0.01,'facealpha',0.5);
    h = mArrow3(vec_x2,vec_y2,'color','blue','stemWidth',0.01,'facealpha',0.5);
    text(-0.4, -.1,'Coordinate 1')

    noiseLevel = 0.04;
    vec_x1 = [0.5 0 0];
    vec_x2 = [0.6 0 0];
    vec_y1 = [0.5 x2(i) 0] + noiseLevel*randn(1, 3);
    vec_y2 = [0.6 x2(i) 0] + noiseLevel*randn(1, 3);
    h = mArrow3(vec_x1,vec_y1,'color','red','stemWidth',0.01,'facealpha',0.5);
    h = mArrow3(vec_x2,vec_y2,'color','blue','stemWidth',0.01,'facealpha',0.5);
    text(0.35, -.1,'Coordinate 2')
    axis([-0.5 1 -0.2 1]);
    axis off
    title('Rotated Data (first two dimensions)');
    drawnow
    % break
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', 0);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0);
    end
end
