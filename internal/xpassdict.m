function dblur = xpassdict(pd, sig, lo),

if ~exist('sig', 'var'),
  sig = 10;
end
if ~exist('lo', 'var'),
  lo = false;
end
  
% build blurred dgray
dblur = zeros(size(pd.dgray));
fil = fspecial('gaussian', [pd.sbin*pd.ny pd.sbin*pd.nx], sig);
for i=1:pd.k,
  elemnorm = norm(pd.dgray(:, i));
  elem = reshape(pd.dgray(:, i), [(pd.ny+2)*pd.sbin (pd.nx+2)*pd.sbin]);
  if lo,
    elemp = filter2(fil, elem, 'same');
  else,
    elemp = elem - filter2(fil, elem, 'same');
  end
  dblur(:, i) = elemp(:) / elemnorm;
end

if nargout == 0, 
  sy = 10;
  sx = 10;
  gny = (pd.ny+2)*pd.sbin;
  gnx = (pd.nx+2)*pd.sbin;
  bord = 10;
  midpad = 1;
  cy = (gny+bord);
  cx = (gnx*2+bord+midpad);
  iii = randperm(size(pd.dgray,2));
  for i=1:min(sy*sx, pd.k),
    row = mod(i-1, sx)+1;
    col = floor((i-1) / sx)+1;

    graypic = pd.dgray(:, iii(i));
    graypic = reshape(graypic, [gny gnx]);
    graypic(:) = graypic(:) - min(graypic(:));
    graypic(:) = graypic(:) / max(graypic(:));

    blurpic = dblur(:, iii(i));
    blurpic = reshape(blurpic, [gny gnx]);
    blurpic(:) = blurpic(:) - min(blurpic(:));
    blurpic(:) = blurpic(:) / max(blurpic(:));

    pic = cat(2, graypic, ones(gny, midpad), blurpic);
    pic = padarray(pic, [bord bord], 1, 'post');

    im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx) = pic;
  end
  im = im(1:end-bord, 1:end-bord);
  imagesc(im);
  axis image;
end
