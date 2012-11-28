figure(1);

subplot(131);
im = double(imread('2007_000272.jpg')) / 255.;
imagesc(im); axis image;
title('Original Image');

subplot(132);
feat = features(im, 8);
showHOG(feat);
title('HOG Features');

subplot(133);
ihog = invertHOG(feat);
imagesc(ihog); axis image;
title('HOG Inverse');
