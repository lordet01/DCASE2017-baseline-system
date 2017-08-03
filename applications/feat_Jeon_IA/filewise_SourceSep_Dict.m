function filewise_SourceSep_Dict(path_in, path_denoise, event_list, event_target)

addpath('src');
addpath('settings');

[~,ftype] = strtok(path_in, '.');

SED_initial_setting_SNMF;


%% Load and organize event bases
K = length(event_list);
n1 = p.F_order * floor(2*p.Splice+1);
n2 = p.F_order * floor(2*p.Splice+1);
B_Mel_x_buff = zeros(n1, p.R_x * (K-1));
lambda_Mel_x_buff = 0;
k_idx = 1;
for k = 1:K
    name = event_list{k};
    load(['dict/event_',name,'/event_',name,'_mel_Dict','.mat']);
    
    if strcmp(event_target, name)
        B_Mel_x = B_event;
        lambda_Mel_x = LAMBDA_g;
    else
        B_Mel_x_buff(:, (k_idx-1)*p.R_x + 1 : k_idx*p.R_x) = B_event;
        
        if iscell(lambda_Mel_x_buff) == 0
            lambda_Mel_x_buff = LAMBDA_g;
        else
% %             lambda_Mel_x_buff = [lambda_Mel_x_buff, LAMBDA_g];
        end
        k_idx = k_idx + 1;
    end
end

B_Lin_x_buff = zeros(n2, p.R_x * (K-1));
lambda_Lin_x_buff = 0;
k_idx = 1;
for k = 1:K
    name = event_list{k};
    load(['dict/event_',name,'/event_',name,'_lin_Dict','.mat']);
    
    if strcmp(event_target, name)
        B_Lin_x = B_event;
        lambda_Lin_x = LAMBDA_g;
    else
        B_Lin_x_buff(:, (k_idx-1)*p.R_x + 1 : k_idx*p.R_x) = B_event;
        
        if iscell(lambda_Lin_x_buff) == 0
            lambda_Lin_x_buff = LAMBDA_g;
        else
% %             lambda_Lin_x_buff = [lambda_Lin_x_buff, LAMBDA_g];
        end
        k_idx = k_idx + 1;
    end
end


%% Load bgn basis
load(['dict/bgn_DCASE2017/bgn_DCASE2017_mel_Dict.mat']);
B_Mel_d = B_event; lambda_Mel_d = LAMBDA_g;
load(['dict/bgn_DCASE2017/bgn_DCASE2017_lin_Dict.mat']);
B_Lin_d = B_event; lambda_Lin_d = LAMBDA_g;

SED_initial_setting_SNMF;

%% Structured Noise Basis Organization {ODL, BGN, NonTarget Event}
if p.Structure_Basis == 1
    B_Mel_d = [B_Mel_d, B_Mel_x_buff]; %lambda_Mel_d = [lambda_Mel_d, lambda_Mel_x_buff];%Consider non-target event as noises
    B_Lin_d = [B_Lin_d, B_Lin_x_buff]; %lambda_Lin_d = [lambda_Lin_d, lambda_Lin_x_buff];%Consider non-target event as noises
end
if p.adapt_train_N == 1
    B_Mel_d = [B_Mel_d(:, 1:p.R_a), B_Mel_d]; %Reserve Noise adaptation field
    B_Lin_d = [B_Lin_d(:, 1:p.R_a), B_Lin_d];
end


%% Clear event-wise buffer
try
    p = rmfield(p, {'w_update_ind', 'h_update_ind', 'init_w', 'init_h'});
catch
end

p.EVENT_NUM = 1;
p.EVENT_RANK = [1];
p.NOISE_NUM = 1;
p.NOISE_RANK = [1];

ch = p.ch;
%Multichannel file I/O initialization
for j = 1:ch
    fin(j) = fopen([path_in(j,:)],'rb');
end

for j = 1:ch
    fdenoise(j) = fopen([path_denoise(j,:)],'wb');
end

frame_len = p.framelength;
frame_shift = p.frameshift;

%% Buffer initialization
if strcmp(p.NMF_algorithm,'NTF') || strcmp(p.NMF_algorithm,'PMWF')  
    g=init_buff_NTF(B_Mel_x, B_Mel_d, B_Lin_x, B_Lin_d, p);
elseif strcmp(p.NMF_algorithm,'SNMF')
    g=init_buff(B_Mel_x, B_Mel_d, B_Lin_x, B_Lin_d, lambda_Mel_x, lambda_Mel_d, lambda_Lin_x, lambda_Lin_d, p);
end

y = zeros(ch, frame_len, 1);
x_tilde = zeros(ch, frame_len,1);

%% Wav processing
if strcmp(ftype,'.wav')
    header = zeros(ch, 44);
    for j = 1:ch
        header(j, :)=fread(fin(j), 44, 'int8'); %Skip wav header parts
    end
end

l = 1;
cnt_residue = 0;
s_in_sub = zeros(ch, frame_shift);
feat_traj = 0;
while (1)
    
    %Check eof
    [~, len] = fread(fin(1), frame_shift, 'bit24');
    
    if cnt_residue > p.delay
        break;
    end
    
    if len ~= frame_shift
        cnt_residue = cnt_residue + 1;
        y = zeros(ch, frame_len, 1);
    else
        fseek(fin(1),-3*frame_shift,0); %Rewind file pointer moved by eof check
        for j = 1:ch
            [s_in_sub(j,:), ~] = fread(fin(j), frame_shift, 'bit24');
            s_in_sub = s_in_sub ./ ((2^23)-1);
            s_in_sub = s_in_sub .* p.input_boost;
        end
        
        %Frame_wise queing
        y(:, 1:frame_len-frame_shift) = y(:, frame_shift+1:frame_len);
        y(:, frame_len-frame_shift+1:frame_len) = s_in_sub;
    end
    
    %Give initial noise flag for basis update
    if l <= p.init_N_len ...
            % && l >= 2*p.Splice + 1
        g.W = 1;
%         g.BETA_blk = g.BETA_blk_init;
    else
        g.W = 0;
%         g.BETA_blk = BETA_blk_tmp;
    end
    
    %Put frame-wise algorithm here
    if strcmp(p.NMF_algorithm,'NTF')
%         [e_est_frame,n_est_frame, d_frame, g] = bntf_sep_event_RT_ASP(y, g, p);
    elseif strcmp(p.NMF_algorithm,'PMWF')
%         [e_est_frame,n_est_frame, d_frame, g] = bntf_sep_event_RT_PMWF(y, g, p);
    elseif strcmp(p.NMF_algorithm,'SNMF')
        [~,~ , d_frame, feat, g] = nmf_sep_event_RT_DCASE17_Dict(y, l, g, p);
    end
    
    for j =1:ch
        if l > p.delay
            x_tilde(j,1:frame_len-frame_shift) = x_tilde(j,frame_shift+1:frame_len);
            x_tilde(j,frame_len-frame_shift+1:frame_len) = zeros(frame_shift,1);
            x_tilde(j,:) = x_tilde(j,:) + d_frame(j,:);
            fwrite(fdenoise(j), x_tilde(j,1:frame_shift).*((2^23)-1), 'bit24');
        end
    end
    
% %     %Store Activation trajectory
% %     if feat_traj == 0
% %         feat_traj = feat;
% %     elseif feat ~= 0
% %         feat_traj = [feat_traj, feat];
% %     end
    
    l = l + 1;
end
% save([path_denoise(1,:),'.mat'], 'feat_traj'); %Store Activation trajectory

fclose('all');

for j = 1:ch
    pcm2wav(path_denoise(j,:), 24, p);
end
