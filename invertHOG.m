% invertHOG(feat)
%
% This function recovers the natural image that may have generated the HOG
% feature 'feat'.

function im = invertHOG(feat),

global ihog_pd

if isempty(ihog_pd),
  load('dict.mat');
end
