%%
hdrfiles = dir(['./*','.hdr']);
mkdir sh
mkdir coeffs_512
% load img2
clear hdrname
c1 = 0.429043;
c2 = 0.511664
c3 = 0.743125
c4 = 0.886227
c5 = 0.247708

load img2
imgall = reshape(img2,[100*200,9]);
imgall(:,1)= imgall(:,1)*c4;
imgall(:,2:4)= imgall(:,2:4)*2*c2;
imgall(:,5:6)= imgall(:,5:6)*2*c1;
imgall(:,7)= imgall(:,7)*c3/3;
imgall(:,8)= imgall(:,8)*2*c1;
imgall(:,9)= imgall(:,9)*c1;
for hdrindex =1:length(hdrfiles)
    cd coeffs_512
    hdrname =hdrfiles(hdrindex).name
hdr = hdrread(['../' hdrfiles(hdrindex).name]);

% hdr0 = imresize(hdr,[900,1800]);
hdr0 = hdr;
hdrname = hdrname(1:length(hdrname)-7)
hdrname = [hdrname '/'];
mkdir(hdrname)
cd ..

for ang =1:512
hdr2= hdr0;
hdr = hdr2;
if ang>1
% hdr(:,1:1800-(ang-1)*50,:) = hdr2(:,1+(ang-1)*50:1800,:);
% hdr(:,1800-(ang-1)*50+1:1800,:) = hdr2(:,1:(ang-1)*50,:);
hdr(:,1+(ang-1)*4:2048,:)=hdr2(:,1:2048-(ang-1)*4,:);
hdr(:,1:(ang-1)*4,:)=hdr2(:,2048-(ang-1)*4+1:2048,:)  ;
end
hdr = hdr(:,2048:-1:1,:);
% figure,imshow(hdr)
hdr = im2double(hdr);
hdr = imresize(hdr,[100,200]);


hdrnew = reshape(hdr,[100*200,3]);
hdr_r = reshape(hdr(:,:,1),[100*200,1]);
red_coeff = imgall\hdr_r;
hdrnew(:,1)=imgall*red_coeff;
hdr_g = reshape(hdr(:,:,2),[100*200,1]);
green_coeff = imgall\hdr_g;
hdrnew(:,2)=imgall*green_coeff;
hdr_b = reshape(hdr(:,:,3),[100*200,1]);
blue_coeff = imgall\hdr_b;
hdrnew(:,3)=imgall*blue_coeff;
hdrnew = reshape(hdrnew,[100,200,3]);

 coeffall = [red_coeff green_coeff blue_coeff];

coeffall(5,:)=-coeffall(5,:);
coeffall(4,:)=-coeffall(4,:);
coeffall(8,:)=-coeffall(8,:);
coeffall


hdrnew(find(hdrnew<0))=0;
tail = ['_' num2str(ang)];
tail = [tail '.txt'];
% hdrname = hdrfiles(hdrindex).name;
% hdrname = hdrname(1:length(hdrname)-7)
% hdrname = [hdrname '/'];

fileID = fopen(strcat(['coeffs_512/' hdrname],[hdrfiles(hdrindex).name tail]),'w');
% fileID = fopen(strcat('coeffs_36/',[hdrfiles(hdrindex).name tail]),'w');
for i =1:9
    for j =1:3
fprintf(fileID,num2str(coeffall(i,j)));
fprintf(fileID,' ');
    end
fprintf(fileID,'\n');
end
fclose(fileID);
end
end