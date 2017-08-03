function [g]=init_buff(B_Mel_x, B_Mel_d, B_Lin_x, B_Lin_d, HMM_Mel_x, HMM_Mel_d, HMM_Lin_x, HMM_Lin_d, p)

addpath('toolbox/LFCC_rastamat');

%set local parameters
[n1,~] = size(B_Mel_x);
[n2,R_x] = size(B_Lin_x);
% n1_unit = floor(n1 /  (2*p.Splice+1));
% n2_unit = floor(n2 /  (2*p.Splice+1));

n_f = n1;
% n_f_unit = n1_unit;

[~,R_d] = size(B_Lin_d);
m = p.blk_len_sep;
sz = p.framelength;

%Global buffer update
g.Ym = zeros(n2,m) + p.nonzerofloor;
g.Yp = zeros(n2,m);

g.Xm_Mel = zeros(p.EVENT_NUM,n1,m);
g.Dm_Mel = zeros(p.NOISE_NUM,n1,m);

g.Xm_hat = zeros(p.EVENT_NUM,n2,m);
g.Dm_hat = zeros(p.NOISE_NUM,n2,m);
g.Xm_tilde = zeros(n_f,m);
g.Xm_sep_1d = zeros(n_f, m);
g.Dm_sep_1d = zeros(n_f, m);

g.x_tilde = zeros(sz, m);
g.x_hat = zeros(p.EVENT_NUM,sz, m);
g.d_hat = zeros(p.NOISE_NUM,sz, m);

g.blk_cnt = 1;
g.B_Mel_x = B_Mel_x;
g.B_Mel_d = B_Mel_d;
g.B_Lin_x = B_Lin_x;
g.B_Lin_d = B_Lin_d;
g.A_d = rand(R_d, m);
g.Ad_blk = rand(p.R_a, p.m_a*m);
g.lambda_d_blk = zeros(n_f, p.m_a*m) + 0.001;
g.lambda_Gy = zeros(n_f,m) + 0.001;
g.update_switch = 1;
g.r_blk = zeros(n_f,p.P_len_l);
g.HMM_Mel_x = HMM_Mel_x;
g.HMM_Mel_d = HMM_Mel_d;
g.HMM_Lin_x = HMM_Lin_x;
g.HMM_Lin_d = HMM_Lin_d; 

%Buff from MCRA technique
g.lambda_dav = zeros(n_f,m);
g.l_mod_lswitch = 0;


%% Make transformation matrix (Mel, Linear)
melmat = fft2melmx(p.fftlength, p.fs, p.F_order, 1, p.DCfreq, p.fhigh, 1, 1);
g.melmat = melmat(:,1:round(p.fftlength*0.5)+1);

params.brkfrq = p.fs/2; % to get linear scale
params.dorasta = 0 ;
linmat = fft2melmxXinhui(p.fftlength, p.fs, p.F_order, 1, p.DCfreq, p.fhigh, params, 0, 1);
g.linmat = linmat(:,1:round(p.fftlength*0.5)+1);

% buff for HMM transition
g.A_buff = zeros(R_x+R_d, p.Patch_Len);
g.Xm_tilde_last = zeros(n_f,1);
g.A_w_Mel_1d = zeros(p.R_Dict,1);
