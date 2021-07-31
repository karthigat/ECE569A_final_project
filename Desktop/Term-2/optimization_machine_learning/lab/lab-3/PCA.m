% function []=PCA(Dtr)
load('X1600');
load('Te28');
load('Lte28');
u=ones(1,1600);
ytr=[u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u];

Dhtr=[X1600;ytr];

m_Dtr=mean(X1600,2);
Xtr=zeros(784,16000);
Xte=zeros(784,10000);
for i=1:length(X1600)
    Xtr(:,i)=X1600(:,i)-m_Dtr;
end
disp(size(Xtr));
c=(Xtr*Xtr')/length(X1600);
disp('step1')
disp(size(c));
[C,s] = eigs(c,330);
disp('step2')
disp(size(C));
X_tr = C'*Xtr;
disp('step3')
disp(size(X_tr));
% X_trn = C*X_tr


%normalize
Xtrn=zeros(330,16000);
m=zeros(1,330);
v=zeros(1,330);
for i=1:330
    xi=X_tr(i,:);
    m(i)=mean(xi);
    v(i)=sqrt(var(xi));
    Xtrn(i,:)=(xi-m(i))/v(i);
end

D_tr = [Xtrn;ytr];
disp('step4')
disp(size(D_tr));

m_Dte=mean(Te28,2);
for i=1:length(Te28)
    Xte(:,i)=Te28(:,i)-m_Dte;
end
disp(size(Xte));
c1=(Xte*Xte')/length(Te28);
disp(size(c1));
[C1,s1] = eigs(c1,330);
disp(size(C1));
X_te = C1'*Xte;
disp(size(X_te));

%normalize
Xten=zeros(330,10000);
m=zeros(1,330);
v=zeros(1,330);
for i=1:330
    xi=X_te(i,:);
    m(i)=mean(xi);
    v(i)=sqrt(var(xi));
    Xten(i,:)=(xi-m(i))/v(i);
end

% X_ten = C*X_te;
D_te = [Xten;1+Lte28(:)'];
disp(size(D_te));
% 
% Mdl2 = fitcsvm(X_tr,ytr,'KernelFunction','linear','Standardize',true);
% disp(Mdl2);
% [label,score] = predict(Mdl2,X_te);
% disp(label);
% disp(score)
x0=zeros(330,16000);
ws_h = GD_Lab3('f_wdbc','g_wdbc',x0,D_tr,0.075,30)
% [ws_h,f]=SRMCC_bfgsML(D_tr,'f_SRMCC','g_SRMCC',0.002,10,62); %hOG dataset
disp(size(ws_h));
disp(ws_h);

Dtest_h=[Xten;ones(1,10000)];
[~,ind_pre]= max((Dtest_h'*ws_h)');
% total_time=cputime-t1+cpth;

C = zeros(10,10);
ytest = 1+Lte28(:)';
for j=1:10
    ind_j=find(ytest == j);
    for i=1:10
        ind_pre_i=find(ind_pre == i);
        C(i,j) = length(intersect(ind_j,ind_pre_i));
    end
end
disp(C);
num_correct=trace(C);
acc=num_correct/sum(C,'all')
