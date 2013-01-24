function vis = hoggles(im),

if length(size(im)) == 2,
  im = repmat(im, [1 1 3]);
end

feat = features(im, 8);

ihog = invertHOG(feat);
ihog = hoggles_process(ihog, size(im), 5);

glyph = HOGpicture(feat);
glyph = hoggles_process(glyph, size(im), 5);

im = padarray(im, [5 5], 0.5);

vis = [im ihog glyph];


function im = hoggles_process(im, size, bord),

im = imresize(im, [size(1) size(2)]);
im(im > 1) = 1;
im(im < 0) = 0;
im = padarray(im, [bord bord], 0.5);
im = repmat(im, [1 1 3]);
