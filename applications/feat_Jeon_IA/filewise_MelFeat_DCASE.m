function filewise_MelFeat_DCASE(path_in)

addpath('src');
addpath('settings');

%Path definition
[file_name, path_mid] = strtok(flip(path_in), '\\');
file_name = flip(file_name);
path_mid = flip(path_mid);
path_out = [path_mid,'/proc']; mkdir(path_out);
path_denoise = [path_out, '/', file_name];


event_list = {'babycry', 'glassbreak', 'gunshot'};
K = strfind(path_in, 'babycry');
if K >= 1
    event_target = 'babycry';
end

K = strfind(path_in, 'glassbreak');
if K >= 1
    event_target = 'glassbreak';
end

K = strfind(path_in, 'gunshot');
if K >= 1
    event_target = 'gunshot';
end

[~,ftype] = strtok(path_in, '.');


SED_initial_setting_SNMF;

if p.train_MLD == 1
    opt_MLD = '_MLD';
else
    opt_MLD = [];
end

%% Load and organize event bases
K = length(event_list);
n1 = p.F_order * floor(2*p.Splice+1);
n2 = p.F_DFT_order * floor(2*p.Splice+1);
B_Mel_x_buff = zeros(n1, p.R_x * (K-1));
B_DFT_x_buff = zeros(n2, p.R_x * (K-1));
k_idx = 1;
for k = 1:K
    name = event_list{k};
    load(['basis/event_',name,'/event_',name,'_Basis',opt_MLD,'.mat']);
    
    if strcmp(event_target, name)
        B_DFT_x = B_DFT_sub; B_Mel_x = B_Mel_sub;
    else
        B_DFT_x_buff(:, (k_idx-1)*p.R_x + 1 : k_idx*p.R_x) = B_DFT_sub; B_Mel_x_buff(:,(k_idx-1)*p.R_x + 1 : k_idx*p.R_x) = B_Mel_sub;
        k_idx = k_idx + 1;
    end
end

%% Load bgn basis
load(['basis/bgn_DCASE2017/bgn_DCASE2017_Basis.mat']);
B_DFT_d = B_DFT_sub; B_Mel_d = B_Mel_sub;

%% Structured Noise Basis Organization {ODL, BGN, NonTarget Event}
B_DFT_d = [B_DFT_d, B_DFT_x_buff]; B_Mel_d = [B_Mel_d, B_Mel_x_buff]; %Consider non-target event as noises
if p.adapt_train_N == 1
    B_DFT_d = [B_DFT_d(:, 1:p.R_a), B_DFT_d]; %Reserve Noise adaptation field
    B_Mel_d = [B_Mel_d(:, 1:p.R_a), B_Mel_d]; %Reserve Noise adaptation field
end

SED_initial_setting_SNMF;

%% Clear event-wise buffer
try
    p = rmfield(p, {'w_update_ind', 'h_update_ind', 'init_w', 'init_h'});
catch
end

p.EVENT_NUM = 1;
p.EVENT_RANK = [1];
p.NOISE_NUM = 1;
p.NOISE_RANK = [1];
p.R_d = p.R_x;
if size(B_DFT_d,2) < p.R_d
    R_tmp = p.R_d - size(B_DFT_d,2);
    B_DFT_d = [B_DFT_d, B_DFT_d(:,1:R_tmp)];
    B_Mel_d = [B_Mel_d, B_Mel_d(:,1:R_tmp)];
end

if strcmp(p.B_sep_mode, 'Mel')
    B1_x = B_Mel_x; B1_d = B_Mel_d;
else
    B1_x = B_DFT_x; B1_d = B_DFT_d;
end
B2_x = B_DFT_x; B2_d = B_DFT_d;

ch = p.ch;

%Multichannel file I/O initialization
for j = 1:ch
    fin(j) = fopen([path_in(j,:)],'rb');
end

% for j = 1:ch
%     for i = 1:p.EVENT_NUM
%         [name, ext] =strtok(path_event(j,:), '.');
%         fevent(i,j) = fopen([name,'_',num2str(i),ext],'wb');
%     end
%     for i =1:p.NOISE_NUM
%         [name, ext] =strtok(path_noise(j,:), '.');
%         fnoise(i,j) = fopen([name,'_',num2str(i),ext],'wb');
%     end
% end

for j = 1:ch
    fdenoise(j) = fopen([path_denoise(j,:)],'wb');
end

frame_len = p.framelength;
frame_shift = p.frameshift;

%% Buffer initialization
if strcmp(p.NMF_algorithm,'NTF') || strcmp(p.NMF_algorithm,'PMWF')  
    g=init_buff_NTF(B1_x, B1_d, B2_x, B2_d, p);
elseif strcmp(p.NMF_algorithm,'SNMF')
    g=init_buff(B1_x, B1_d, B2_x, B2_d, p);
end

y = zeros(ch, frame_len, 1);
% d_hat = zeros(p.NOISE_NUM, ch, frame_len, 1);
% x_hat = zeros(p.EVENT_NUM, ch, frame_len, 1);
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
        [~,~ , d_frame, feat, g] = nmf_sep_event_RT_DCASE17(y, l, g, p);
    end
    
    for j =1:ch
        if l > p.delay
%             for i = 1: p.NOISE_NUM
%                 d_hat(i,j,1:frame_len-frame_shift) = d_hat(i,j,frame_shift+1:frame_len);
%                 d_hat(i,j,frame_len-frame_shift+1:frame_len) = zeros(frame_shift,1);
%                 d_hat(i,j,:) = d_hat(i,j,:) + n_est_frame(i,j,:);
%                 s_noise_sub = d_hat(i,j,1:frame_shift);
%                 fwrite(fnoise(i,j), s_noise_sub, 'int16');
%             end
%             
%             for i = 1: p.EVENT_NUM
%                 x_hat(i,j,1:frame_len-frame_shift) = x_hat(i,j,frame_shift+1:frame_len);
%                 x_hat(i,j,frame_len-frame_shift+1:frame_len) = zeros(frame_shift,1);
%                 x_hat(i,j,:) = x_hat(i,j,:) + e_est_frame(i,j,:);
%                 s_event_sub = x_hat(i,j,1:frame_shift);
%                 fwrite(fevent(i,j), s_event_sub, 'int16');
%             end
            x_tilde(j,1:frame_len-frame_shift) = x_tilde(j,frame_shift+1:frame_len);
            x_tilde(j,frame_len-frame_shift+1:frame_len) = zeros(frame_shift,1);
            x_tilde(j,:) = x_tilde(j,:) + d_frame(j,:);
            fwrite(fdenoise(j), x_tilde(j,1:frame_shift).*((2^23)-1), 'bit24');
        end
    end
    
    %Store Activation trajectory
    if feat_traj == 0
        feat_traj = feat;
    else
        feat_traj = [feat_traj, feat];
    end
    
    l = l + 1;
end
mel_spectrum = feat_traj ./ max(max(feat_traj)) .* 0.9;
mel_spectrum = mel_spectrum(:,1:end-2);
save(['./tmp/tmp_melfeat.mat'], 'mel_spectrum'); %Store Activation trajectory

fclose('all');

for j = 1:ch
    pcm2wav(path_denoise(j,:), 24, p);
end

quit;
