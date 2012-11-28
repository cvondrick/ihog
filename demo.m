figure(1);

subplot(131);
im = double(imread('2007_000272.jpg')) / 255.;
imagesc(im); axis image;

subplot(132);
feat = features(im, 8);
showHOG(feat);

subplot(133);
ihog = invertHOG(feat);
imagesc(ihog); axis image;
