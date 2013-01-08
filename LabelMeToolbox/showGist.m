function showGist(gist, param)
%
% Visualization of the gist descriptor
%   showGist(gist, param)
%
% The plot is color coded, with one color per scale
%
% Example:
%   img = zeros(256,256);
%   img(64:128,64:128) = 255;
%   gist = LMgist(img, '', param);
%   showGist(gist, param)


[Nimages, Ndim] = size(gist);
nx = ceil(sqrt(Nimages)); ny = ceil(Nimages/nx);

Nblocks = param.numberBlocks;
Nfilters = sum(param.orientationsPerScale);
Nscales = length(param.orientationsPerScale);

C = hsv(Nscales);
colors = [];
for s = 1:Nscales
    colors = [colors; repmat(C(s,:), [param.orientationsPerScale(s) 1])];
end
colors = colors';

[nrows ncols Nfilters] = size(param.G);
Nfeatures = Nblocks^2*Nfilters;

if Ndim~=Nfeatures
    error('Missmatch between gist descriptors and the parameters');
end

G = param.G(1:2:end,1:2:end,:);
[nrows ncols Nfilters] = size(G);
G = G + flipdim(flipdim(G,1),2);
G = reshape(G, [ncols*nrows Nfilters]);


if Nimages>1
    figure;
end

for j = 1:Nimages
    g = reshape(gist(j,:), [Nblocks Nblocks Nfilters]);
    g = permute(g,[2 1 3]);
    g = reshape(g, [Nblocks*Nblocks Nfilters]);
           
    for c = 1:3
        mosaic(:,c,:) = G*(repmat(colors(c,:), [Nblocks^2 1]).*g)';
    end
    mosaic = reshape(mosaic, [nrows ncols 3 Nblocks*Nblocks]);    
    mosaic = fftshift(fftshift(mosaic,1),2);
    mosaic = uint8(mosaic/max(mosaic(:))*255);
    mosaic(1,:,:,:) = 255;
    mosaic(end,:,:,:) = 255;
    mosaic(:,1,:,:) = 255;
    mosaic(:,end,:,:) = 255;
    
    if Nimages>1
        subplottight(ny,nx,j,0.01);
    end
    montage(mosaic, 'size', [Nblocks Nblocks])
end
