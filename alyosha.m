function im = alyosha(im, pd),

feat = features(im, 8);
ihog = invertHOG(feat, pd);

im = imresize(im, [size(ihog, 1) size(ihog, 2)]);
im(im > 1) = 1;
im(im < 0) = 0;

origalt = rgb2hsv(im);

alt = rgb2hsv(ihog);
alt(:, :, 3) = origalt(:, :, 3);

im = hsv2rgb(alt);

imagesc(im); axis image;
