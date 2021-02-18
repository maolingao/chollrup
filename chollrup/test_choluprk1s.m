% benchmark choluprk1.c and choluprk1s.c
% require muldiag() in git@github.com:maolingao/essential.git
%
path = matlab.desktop.editor.getActiveFilename;
path = fileparts(path);
cd(path)
addpath('../../essential/essential');
%
n=1e3;
maxlam=2; minlam=0.1;
cvecs=zeros(n,1); svecs=zeros(n,1);
cvecf=zeros(n,1); svecf=zeros(n,1);
wkvecf=zeros(3*n,1);
k=10;
ts = nan(k,1); tf = nan(k,1);
errs = nan(k,1); errf = nan(k,1);

%% test sparse positive update
fprintf(1,'Testing minimum pos. updates\n');

fprintf("nnz(full_factor)=%d.\n",n+(n^2-n)/2);
for i=1:k
    % -- random problem generation -- %
    ah = sprand(n,n,0.1);
    a = ah*ah'+1e0*speye(n);
    lfacts = chol(a,'lower');
    %
    ltmp=full(lfacts);
    lfactf=ltmp(:,:);    % !force copy!
    vec=randn(n,1);
    vec_copy = vec(:,:); % !force copy!
    % expected updated factor
    l_expect=chol(a+vec*vec','lower');

    % -- sparse variant -- %
    t=tic;
    if choluprk1s(lfacts,vec,cvecs,svecs)~=0
    error('Numerical error in CHOLUPRK1s!');
    end
    ts(i)=toc(t);
    errs(i)=max(max(abs(lfacts-l_expect)));

    % -- dense variant -- %
    t=tic;
    if choluprk1({lfactf,[1 1 n n],'L '},vec_copy,cvecf,svecf,wkvecf)~=0
    error('Numerical error in CHOLUPRK1!');
    end
    tf(i)=toc(t);
    errf(i)=max(max(abs(lfactf-l_expect)));
    
    fprintf(1,'[choluprk1s] Max. dist. L: %e, runtime=%3.2e, nnz(lfact)=%d.\n',errs(i), ts(i), nnz(lfacts)); 
    fprintf(1,'[choluprk1 ] Max. dist. L: %e, runtime=%3.2e.\n',errf(i), tf(i));
end

fprintf("[summary] average runtime: choluprk1s=%3.2e, choluprk1=%3.2e.\n",mean(ts),mean(tf));
fprintf("[summary] average accuracy: choluprk1s=%3.2e, choluprk1=%3.2e.\n",mean(errs),mean(errf));

%% result
%{
% 2021-02-18
n=10; k=10;
[summary] average runtime: choluprk1s=1.16e-04, choluprk1=1.19e-04.
[summary] average accuracy: choluprk1s=4.00e-16, choluprk1=4.00e-16.

n=1e2; k=10;
[summary] average runtime: choluprk1s=2.01e-04, choluprk1=1.10e-04.
[summary] average accuracy: choluprk1s=1.08e-15, choluprk1=1.08e-15.

n=1e3; k=10;
[summary] average runtime: choluprk1s=4.73e-03, choluprk1=5.31e-04.
[summary] average accuracy: choluprk1s=5.59e-07, choluprk1=8.66e-15.

n=1e4; k=10;
[summary] average runtime: choluprk1s=1.02e+00, choluprk1=8.13e-02.
[summary] average accuracy: choluprk1s=9.97e-07, choluprk1=2.37e-13.

n=1e5; k=3;

%}