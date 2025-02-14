%Embedded Knowledge Transfer for Brain-Computer Interfaces (MEKT)
% =====================
% Author: Wen Zhang and Dongrui Wu
% Date: Oct. 9, 2019
% E-mail: wenz@hust.edu.cn

clc;
clear all;
close all;
warning off;

% Load datasets: 
% 9 subjects, each 22*750*144 (channels*points*trails)
root='.\data\MI2-1\';
listing=dir([root '*.mat']);
addpath('lib');
addpath(genpath('./util/'));
addpath('.\SCSL-master');
% Load data and perform congruent transform
fnum=length(listing);
Ca=nan(22,22,144*fnum);
Xr=nan(22,750,144*9);
Xa=nan(22,750,144*9);
Y=nan(144*fnum,1);
ref={'riemann','logeuclid','euclid'};

for f=1:fnum
    load([root listing(f).name])
    idf=(f-1)*144+1:f*144;

    Y(idf) = y+1; Xr(:,:,idf) = x;
    Ca(:,:,idf) = centroid_align(x,ref{2});
    [~,Xa(:,:,idf)] = centroid_align(x,ref{2});
end
aaaa=zeros(11,11);j=0;
for dim=[1, 2, 3, 4, 5]
    j=j+1;i=0;
    for dimm=[1, 5, 10, 15, 20]
    i=i+1;
    disp(dim);
N=1; bca_dte=[];

% tic
for t=1:N
%     tic
    BCA=zeros(fnum,1);
    for n=1:fnum
        disp(n)
        % Single target data & multi source data
        idt=(n-1)*144+1:n*144;
        ids=1:144*fnum; ids(idt)=[];             
        Yt=Y(idt); Ys=Y(ids);
        idsP=Yt==1; idsN=Yt==2;
        Ct=Ca(:,:,idt);  Cs=Ca(:,:,ids);
        


        Ft=Xa(:,:,idt);  Fs=Xa(:,:,ids);
        K = dim;
        tau = dimm;
        [R_train, Wh] = Enhanced_cov_train(Fs, K, tau);
        R_test = Enhanced_cov_test(Ft, K, tau, Wh);
        Xs1 = R_train';
        Xt1 = R_test';
        Xs2=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
        Xt2=logmap(Ct,'MI');    
        Xs = [Xs1; Xs2];
        Xt = [Xt1; Xt2];
        
        
        options= defaultOptions(struct(),...
                'T',5,...              % The iteration times
                'dim',30,...            % The dimension of the projection subspace
                'alpha',0.1,...         % The weight of manifold regularization
                'beta',5,...            % The weight of discrimination
                'sC',2,...             % The fuzzy number
                'kernel_type',3,...     % Kernel
                'gamma',1,...           % The hyper-parameter of Kernel
                'lambda',1,...
                'eta',0.1,...
                 't1',0.55,...
                 't2',0.55);            % The regularization term
        
        [acc,acc_ite,max_acc,~,func]=MCPSCSL(Xs,Ys,Xt,Yt,options);
        BCA(n) = max_acc;
        

    end
    disp(mean(BCA)*100)
    bca_dte=[bca_dte,mean(BCA)*100];

end
   
    aaaa(j,i)=mean(mean(BCA,1),2)*100';
    end
end
% toc
rmpath('lib');
