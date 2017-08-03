global p;

%% NMF algorithm options
% p.NMF_algorithm = 'NTF';
% p.NMF_algorithm = 'PMWF'; %PMWF with NE based on second order statistics
p.NMF_algorithm = 'SNMF';
p.useGPU = 0;
p.ForceRewrite = 0; %Forcely rewrite output file
p.ForceRetrain = 0; %Forcely retrain bases
p.ForceRetrain_MLD = 0; %Forcely retrain bases only for MLD
p.ForceRetrain_HMM = 0;
p.ProcBypass = 0;

%% NMF parameters
p.Patch_Len = 12; %Unit spectral patch size <frame> 
p.Patch_shift = 10; %Shift size of unit spectral patch during training
p.R_c = 6; %Unit basis rank for a dictionary
p.R_Dict = 20; %Number of Dictionary for each event
p.R_x = p.R_c * p.R_Dict;
p.R_d = p.R_c * p.R_Dict;
p.nonzerofloor = 1e-9;
p.blk_len_sep=p.Patch_Len;
p.blk_hop_sep=p.Patch_shift;
p.Splice = 0;

%% Signal Parameters
p.fs = 44100;
p.wintime = 0.040;
p.hoptime = 0.020;
p.ch = 1;
p.framelength =round(p.wintime*p.fs);
p.frameshift  =round(p.hoptime*p.fs);
p.delay = p.Splice + p.blk_len_sep + floor(p.wintime / p.hoptime / 2 + 0.5); %Delay compensated output
% p.delay = 0; %Delayed Output
p.fftlength = 2^ceil(log2(p.framelength));     
p.F_DFT_order = 0.5*p.fftlength+1;
p.F_order = 64;
p.overlapscale = 2*p.frameshift/p.framelength;
p.win_STFT = sqrt(hann(p.framelength,'periodic')); %Set type of window
p.win_ISTFT = sqrt(hann(p.framelength,'periodic')); %Set type of window
p.pow = 2; %Power coefficient: [1: Mag., 2: Pow]
p.input_boost = 32; %Boost input signal (only for DCASE data)


%% Basis Training parameters (Mixture of Learned Dictionary, MLD, Sig.Proc. Lett. 15, M. Kim)
p.train_Exemplar = 0; %train with SNMF if 0
p.train_MLD = 1; %Retrain Bases with MLD
p.cluster_buff = 1; %Maximum rank scale before clustering (1: Turn off clustering)
p.train_seq_len_max = ceil(p.fs * 720); %12 min
p.load_file_num = 130; %set only for debug, 0 for load every files in a folder
p.sil_len = 0.5; %minimum length of silence between sub-event. (s)
p.Structure_Basis = 0; %Set Non-target event basis to Noise group 

%% HMM Training Parameters
p.train_HMM_NMF = 1;
p.Q = 3; %Number of state per a event
p.M = 4; %Number of Gaussian Mixture
p.LeftToRight = 1; %0: Ergodic, 1: Left to Right
% % p.LearnDur = p.R_c * p.hoptime; %Learn duration for HMM
% % p.SampleForTrain = 2000;

%% Basis update option (Online Dictionary Learning, ODL, Interspeech 16, K. M. Jeon)
p.adapt_train_N = 1;
p.init_N_len = p.Patch_Len; %No. of initial frames used for nosie basis update
p.R_a = floor(0.5 * p.R_d);
p.m_a = p.Patch_Len*2; %No. of stacked block for basis adaptation
p.overlap_m_a = 0.1; %Update cycle for noise learning
p.Ar_up = 1.0; %Define Ax and Ad ratio for noise dictionary update (Higher: Update frequently, Lower: Update rarely)
p.B_D_u_name = 'B_D_u.mat';
p.basis_update_N = 0;
p.basis_update_E = 0;
p.R_semi = floor(0.1 * p.R_d);
 
%% Sparsity-Based Similarity Check between Input and Basis (17, K. M. Jeon)
p.SparseCheck = 0;
p.SC_RatioL = 0.1;
p.SC_RatioH = 0.95;
p.SC_pow = 2;

%% HMM-Based time transition Check from the past N frames (17, K. M. Jeon)
p.TransitionCheck = 1;
p.TC_pow = 2;

%% Block sparsity options (Block Sparsity Measure for ODL, DSP 17, K. M. Jeon)
p.blk_sparse = 0; %block sparsity switch
p.P_len_k = 8; % vertical (frequency bin) size of Block for local sparsity calculation
p.P_len_l = 6; % horizental (time frame index) size of Block for local sparsity calculation
% p.kappa = 1.0;
p.nu = 1.0;
p.alpha_p = 0.0; %DD smoothing factor for P
p.beta_p = 0.0; %DD smoothing factor for P
p.blk_gap = 1; %Blk_gap for complexity issue (1 for ideal), Odd only!
p.Q_sig_c = 0.3; %Sigmoid parameter c %Set 0.4 for figure
p.Q_sig_L = 1.0; %Max scale for LC_est in SBM
p.Q_sig_a = 20; %Sigmoid parameter a
p.regul = 0.5;


%MDI options
p.MDI_est = 0; %MDI estimation option
p.MDI_est_noise = 0; %MDI estimation option (noise)
p.sparsity_mdi = 5;
p.conv_eps_mdi = 1e-5; 


%PMWF options
p.PMWF = 0;
p.BETA_PMWF = 10; %0: MVDR, >0: PMWF
p.M_PMWF = 2; %spectral neighbor region
p.L_PMWF = 2; %temporal neighbor region
p.ALPHA_E_PMWF = 0.3; %Mixing ratio between current Event PSD-CM vs average one
p.norm_period = p.init_N_len; %Normalization period of PSD cov. matrix
p.Ncov_update = 1;%Update Ncov by cov. mat. from output - input


%Front-end & %Back-end processing
p.preemph = 0.0; %0.92
p.DCfreq = 80; %Hz
p.fhigh = 15000; %Hz
p.DCbin = floor(p.DCfreq / (p.fs / p.fftlength) + 0.5); %Forcely give 0 to 1~N bin which is not important in speech
p.DCbin_back = p.DCbin; %Forcely give 0 to 1~N bin which is not important in speech


%Multi-channel options
p.filegap = p.ch; %No. of file consisting one session


%Training options 
p.separation = 1;
p.B_sep_mode = 'Mel'; %['Lin', 'Mel']
p.MelConv = 1; %Use frequency scale conversion. 00: Coupled dictionary
p.MelOut = 0; %Use Mel spectrum at synthesizing wavform
p.train_VAD = 0;
p.train_ANOT = 0;


%SNMF parameters
p.NMF_GPU = 0;
p.cf = 'kl';   %  'is', 'kl', 'ed'; takes precedence over setting the beta value
p.sparsity = 5; %Activation sparsity constraint
p.sparsity_eta = 10000; %Cone clustering constraint (Higher: Compacter cone size)
p.sparsity_epsilon = 0.00001; %Regularizer that suppress irrelavant cone
p.max_iter = 100; % Stopping criteria
p.conv_eps = 1e-3; 
p.display   = 0; % Display evolution of objective function
p.random_seed = 1; % Random seed: any value over than 0 sets the seed to that value
p.cost_check = 1;
p.est_scale = 1.0;


%Single channel enhancement options
p.ENHANCE_METHOD = 'MMSE'; %['Wiener', 'MMSE']
%2.1) Parameters of "Decision-Directed" a Priori SNR Estimate
p.alpha_eta=0.3;	% Recursive averaging parameter
p.eta_min=10^(-18/10);	% Lower limit constraint
p.alpha_d = 0.95; % Recursive averaging parameter for the noise
p.G_floor = 0.00006;
p.G_beta = 1.0; %Bias compensation factor in the noise estimator
p.G_beta_max = 1000.0;

%Phase compensation options
p.phase_comp = 0;
p.pc_alpha = 0.3; %pc lambda scaling factor 1
p.pc_lambda = 100.0; %Empirical lambda set by K. Wojciki

%VAD for speech training
p.speech_train_start = round(p.fs * 0.5);
p.speech_train_end = round(p.fs * 1.5);
p.speech_train_len = p.speech_train_end - p.speech_train_start;


%Name of the test system
Testname = ['DCASE'];
OUTname = [Testname,'_',p.NMF_algorithm,'_A',num2str(p.adapt_train_N),'_M',num2str(p.MDI_est_noise),'_r',num2str(p.R_x),'_p',num2str(p.pow), ...
           '_',p.ENHANCE_METHOD,'_P',num2str(p.blk_sparse),'_SNMF'];