% diagnoseHOG(feat)
%
% Produces a figure with a large number of HOG visualizations for easier 
% diagnosis of HOG.
function diagnoseHOG(feat),

clf;

subplot(221);
imagesc(visualizeHOG(feat));
axis image;
title('HOG Visualization');

subplot(222);
imagesc(visualizeHOG(feat - mean(feat(:))));
axis image;
title('Zero Mean');

subplot(223);
imagesc(dissectHOG(feat));
axis image;
title('Full, Unsigned, Signed, Texture, PosSign, NegSign');

subplot(224);
imagesc(spreadHOG(feat));
axis image;
title('Individual Channels');
