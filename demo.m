im = double(imread('2007_000272.jpg')) / 255.;
feat = features(im, 8);
ihog = invertHOG(feat);

figure(1);
clf;

subplot(131);
imagesc(im); axis image; axis off;
title('Original Image', 'FontSize', 20);

subplot(132);
showHOG(feat); axis off;
title('HOG Features', 'FontSize', 20);

subplot(133);
imagesc(ihog); axis image; axis off;
title('HOG Inverse', 'FontSize', 20);
