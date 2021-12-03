clc;clear;close all;
%%
class_0=0;
class_1=0;
class_2=0;
class_3=0;
class_4=0;
class_01=0;
class_02=0;
class_03=0;
class_04=0;
class_12=0;
class_13=0;
class_14=0;
class_23=0;
class_24=0;
class_34=0;
class_012=0;
class_013=0;
class_014=0;
class_023=0;
class_024=0;
class_034=0;
class_123=0;
class_124=0;
class_134=0;
class_234=0;
class_1234=0;
True=1;
%%
folderpath='/home/arun/Documents/PyWSPrecision/datasets/Defects_CleanedData_YoloAnnotationFormat/';
addpath(folderpath)
% list all txt files in the path
cd(folderpath)
listing=dir('*.txt*');
class_label_lis=listing(end);
listing(end)=[];
listlen=length(listing);
total_cat_cell=cell(listlen,1);

% listindex=1;
for listindex=1:listlen

% filename='20200917_083154.txt';
filename=listing(listindex).name;
% Path generation
filepath=strcat(folderpath,'/',filename);

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
fclose(fileID);

total_category=zeros(sizA(2),1);
for cati=1:sizA(2)
total_category(cati)=A(1,cati);
end
category=unique(total_category);
if isequal(category,0)==True
    class_0=class_0+1;
elseif isequal(category,1)==True
    class_1=class_1+1;
elseif isequal(category,2)==True
    class_2=class_2+1;
elseif isequal(category,3)==True
    class_3=class_3+1;
elseif isequal(category,4)==True
    class_4=class_4+1;
elseif isequal(category,[0;1])==True
    class_01=class_01+1;
elseif isequal(category,[0;2])==True
    class_02=class_02+1;
elseif isequal(category,[0;3])==True
    class_03=class_03+1;
elseif isequal(category,[0;4])==True
    class_04=class_04+1;
elseif isequal(category,[1;2])==True    
    class_12=class_12+1;
elseif isequal(category,[1;3])==True     
    class_13=class_13+1;
elseif isequal(category,[1;4])==True  
    class_14=class_14+1;
elseif isequal(category,[2;3])==True  
    class_23=class_23+1;
elseif isequal(category,[2;4])==True      
    class_24=class_24+1;
elseif isequal(category,[3;4])==True  
    class_34=class_34+1;
elseif isequal(category,[0;1;2])==True  
    class_012=class_012+1;
elseif isequal(category,[0;1;3])==True  
    class_013=class_013+1;
elseif isequal(category,[0;1;4])==True  
    class_014=class_014+1;
elseif isequal(category,[0;2;3])==True  
    class_023=class_023+1;
elseif isequal(category,[0;2;4])==True  
    class_024=class_024+1;
elseif isequal(category,[0;3;4])==True  
    class_034=class_034+1;
elseif isequal(category,[1;2;3])==True  
    class_123=class_123+1;
elseif isequal(category,[1;2;4])==True  
    class_124=class_124+1;
elseif isequal(category,[1;3;4])==True      
    class_134=class_134+1;
elseif isequal(category,[2;3;4])==True      
    class_234=class_234+1;
else
    class_1234=class_1234+1;
end
    

total_cat_cell{listindex,1}=category;
end
%% Data Visualisation

classes_bar=[class_0;
class_1;
class_2;
class_3;
class_4;
class_01;
class_02;
class_03;
class_04;
class_12;
class_13;
class_14;
class_23;
class_24;
class_34;
class_012;
class_013;
class_014;
class_023;
class_024;
class_034;
class_123;
class_124;
class_134;
class_234;
class_1234;];

classes_bar_cell={'0';
'1';
'2';
'3';
'4';
'01';
'02';
'03';
'04';
'12';
'13';
'14';
'23';
'24';
'34';
'012';
'013';
'014';
'023';
'024';
'034';
'123';
'124';
'134';
'234';
'1234';};
%%
classes_bar_tick={0;
1;
2;
3;
4;
01;
02;
03;
04;
12;
13;
14;
23;
24;
34;
012;
013;
014;
023;
024;
034;
123;
124;
134;
234;
1234;};

classes_bar_tick1=1:26;

% txt = '\leftarrow sin(\pi) = 0';
txt = {'0-Water Stains','1-Mould','2-Electrical Switches & Sockets','3-Cracked Plaster','4-Damaged Silicone'};

% class_filename=class_label_lis.name;
% % Path generation
% class_filepath=strcat(class_label_lis.folder,'/',class_label_lis.name);
% 
% % Open text file in a particular format that enables array processing
% class_fileID = fopen(class_filepath,'r');
% sizeA = [5 Inf];%
% formatSpec_class='%s';
% % Reding the files
% A_class = fscanf(class_fileID,formatSpec_class);


figure(1);
% stem(classes_bar);
bar(classes_bar);xticks(classes_bar_tick1);xticklabels(classes_bar_cell);
text(17,200,txt)