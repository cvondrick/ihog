% pickapartHOG(feat)
%
% This function shows the HOG inverse for different HOG channels: unsigned
% gradients, signed gradients, and texture gradients. 
function pickapartHOG(feat),

fprintf('ihog: pick apart: ');

fprintf('.');
ihog = invertHOG(feat);

fprintf('.');
unsigned = zeros(size(feat));
unsigned(:, :, 19:27) = feat(:, :, 19:27);
unsigned = invertHOG(unsigned);

fprintf('.');
signed = zeros(size(feat));
signed(:, :, 1:18) = feat(:, :, 1:18);
signed = invertHOG(signed);

fprintf('.');
positive = zeros(size(feat));
positive(:, :, 1:9) = feat(:, :, 1:9);
positive = invertHOG(positive);

fprintf('.');
negative = zeros(size(feat));
negative(:, :, 10:19) = feat(:, :, 10:19);
negative = invertHOG(negative);

fprintf('.');
texture = zeros(size(feat));
texture(:, :, 28:end) = feat(:, :, 28:end);
texture = invertHOG(texture);

fprintf('\n');

subplot(231);
imagesc(ihog);
axis image;
title('Full HOG');

subplot(232);
imagesc(unsigned);
axis image;
title('Unsigned HOG');

subplot(233);
imagesc(signed);
axis image;
title('Signed HOG');

subplot(234);
imagesc(texture);
axis image;
title('Texture HOG');

subplot(235);
imagesc(positive);
axis image;
title('Positive Signed HOG');

subplot(236);
imagesc(negative);
axis image;
title('Negative Signed HOG');
