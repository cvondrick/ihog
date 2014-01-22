% Returns a matrix w and vector mu that whitens a ny-by-nx HOG.
%
% Whitening:
%   y = w * (x - mu)
% Coloring:
%   x = c * y + mu;
%
% Adapted from:
% B. Hariharan, J. Malik, D. Ramanan. "Discriminative Decorrelation for
% Clustering and Classification" European Conference on Computer Vision
% (ECCV), Florence, Italy, Oct. 2012.
function [w, mu, c] = getwhiteningmatrix(ny, nx),

[sig, mu] = hogcovariance(ny, nx);
[v,d] = eig(sig);

sqr = sqrt(d);

% compute whitening matrix:
% since d is diagonal, we can efficiently invert 
w = v * diag(1./diag(sqr)) * v';

% compute coloring matrix:
c = v * sqr * v';



function [sig,neg] = hogcovariance(ny,nx,bg);

if ~exist('bg', 'var'),
  load(sprintf('%s/bg.mat', fileparts(mfilename('fullpath'))));
end

neg = repmat(bg.neg',ny*nx,1);
neg = neg(:);
sig = reconstructSig(nx,ny,bg.cov,bg.dxy);
sig = sig + bg.lambda*eye(size(sig));

% add occlusion feature back
sig = blkdiag(sig, eye(ny*nx));
neg = padarray(neg, [ny*nx 0], 0, 'post');
  


function w = reconstructSig(nx,ny,ww,dxy)
% W = reconstructSig(nx,ny,ww,dxy)
% W = n x n 
% n = ny * nx * nf

k  = size(dxy,1);
nf = size(ww,1);
n  = ny*nx;  
w  = zeros(nf,nf,n,n);

for x1 = 1:nx,
  for y1 = 1:ny,
    i1 = (x1-1)*ny + y1;
    for i = 1:k,
      x = dxy(i,1);
      y = dxy(i,2);
      x2 = x1 + x;        
      y2 = y1 + y;
      if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny,
        i2 = (x2-1)*ny + y2;
        w(:,:,i1,i2) = ww(:,:,i); 
      end
      x2 = x1 - x;        
      y2 = y1 - y;
      if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny,
        i2 = (x2-1)*ny + y2; 
        w(:,:,i1,i2) = ww(:,:,i)'; 
      end
    end
  end
end

% Permute [nf nf n n] to [n nf n nf]
w = permute(w,[3 1 4 2]);
w = reshape(w,n*nf,n*nf);

% Make sure returned matrix is close to symmetric
assert(sum(sum(abs(w - w'))) < 1e-5);

w = (w+w')/2;
