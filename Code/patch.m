%Extracting the patches

img=imread('flood.png');
dd=0;
for ii=1:20:size(img,1)
    for jj=1:20:size(img,2)
        imm=img(ii:ii+19,jj:jj+19);
        imwrite(imm,['TestFlood/',num2str(dd),'.jpg']);
        dd=dd+1;
    end
end
