clc;clear;close all;
%%
folderpath='/home/arun/Documents/PyWSPrecision/datasets/Defects_CleanedData_YoloAnnotationFormat';
% list all txt files in the path
% choose one image and text file name to test.


filename='20200917_083154.txt';
filename1='20200917_083154.jpg';
% Path generation
filepath=strcat(folderpath,'/',filename);
filepath1=strcat(folderpath,'/',filename1);

% Open text file in a particular format that enables array processing
fileID = fopen(filepath,'r');
formatSpec = '%f'; % float representation picks the decimal numbers - 
                    % relative bounding box position and size
sizeA = [5 Inf];
% Reding the files
A = fscanf(fileID,formatSpec,sizeA);
% Estimating size of the array for loop wise operation
sizA=size(A);
% reading image
I=imread(filepath1);
% Required in matlab check in with python
I=imrotate(I,-90);
% To change relative postion and size into pixel dimension values
sizI=size(I);

% label_index=2;
subimg=cell(sizA(2),1);
for label_index=1:sizA(2)
    % ATTENTION: xc and yc are not the top left corners like the usual but bounding box's centre
xc=round(A(2,label_index)*sizI(1));
yc=round(A(3,label_index)*sizI(2));
xwidth=round(A(4,label_index)*sizI(1));
yheight=round(A(5,label_index)*sizI(2));

xmin=xc-(xwidth*0.5);
xmax=xc+(xwidth*0.5);

ymin=yc-(yheight*0.5);
ymax=yc+(yheight*0.5);

% Manual cropping / array slicing - datatype reassign maybe needed check in
% python
subimg{label_index,1}=I(xmin:xmax,ymin:ymax,:);
%class extraction
class_label=A(2,label_index);
% 1. check if folder exists if not create one in class label's name
% 2. save sub images in respective class folder

% I2=I(ymin:ymax,xmin:xmax,:);
end
% Repeat the above procedure as a function for all the images through loop.