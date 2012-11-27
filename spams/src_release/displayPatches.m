function tmp = displayPatches(D)

V=1;
[n K] = size(D);
size(D);
sizeEdge=sqrt(n/V);
if floor(sizeEdge) ~= sizeEdge
   V=3;
   sizeEdge=sqrt(n/V);
end

% for (ii = 1:size(D,2))
%    if (D(1,ii) > 0)
%       D(:,ii) = -D(:,ii);
%    end
% end

p=4.5;
        M=max((D(:)));
        m=min((D(:)));
        if (m >= 0)
           me=0;
           sig=sqrt(mean(((D(:))).^2));
        else
           me=mean(D(:));
           sig=sqrt(mean(((D(:)-me)).^2));
        end
        D=D-me;
        D=min(max(D,-p*sig),p*sig);
        M=max((D(:)));
        m=min((D(:)));
       D=(D-m)/(M-m);
  %     D=1-D;

nBins=floor(sqrt(K));

tmp = zeros((sizeEdge+1)*nBins+1,(sizeEdge+1)*nBins+1,V);
patch = zeros(sizeEdge,sizeEdge,1);
mm=sizeEdge*sizeEdge;
for ii = 1:nBins
   for jj = 1:nBins
      io=ii;
      jo=jj;
      offsetx=0;
      offsety=0;
      ii=mod(ii-1+offsetx,nBins)+1;
      jj=mod(jj-1+offsety,nBins)+1;
      patchCol=(D(1:n,(io-1)*nBins+jo));
      patchCol=reshape(patchCol, [sizeEdge,sizeEdge V]);

      M=max((patchCol(:)));
      m=min((patchCol(:)));
  %    patchCol=1.0*(patchCol-m)/(M-m)+0.0;


      tmp((ii-1)*(sizeEdge+1)+2:ii*(sizeEdge+1),...
         (jj-1)*(sizeEdge+1)+2:jj*(sizeEdge+1),:)=patchCol;
      ii=io;
      jj=jo;
   end
end

colormap('bone');
imagesc(tmp);


