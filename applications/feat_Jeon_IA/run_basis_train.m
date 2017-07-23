function [B_DFT, B_Mel, A_DFT, A_Mel, p_out] = run_basis_train(dir_Basis, DB_path, DC_freq_set, VAD_set, event_num, B_name_full, R, p)

addpath('src');
addpath(genpath('toolbox/yaml')); 

B_DFT = zeros(p.F_DFT_order * (2*p.Splice+1), R * event_num);
B_Mel = zeros(p.F_order * (2*p.Splice+1), R * event_num);
DC_bin_set = floor(DC_freq_set ./ (p.fs / p.fftlength) + 0.5);
s_sil = zeros(round(p.fs * p.sil_len), 1);

A_DFT = 0;
A_Mel = 0;
for l = 1:event_num
    A_DFT_tot = 0;
    A_Mel_tot = 0;
    fexist = fopen([dir_Basis, '/',B_name_full{l},'/', B_name_full{l},'_Basis.mat']);
    if fexist == -1 || p.ForceRetrain
        disp(['------',num2str(l),'-th event train------']);
        
        %% Make a full training sequence
        EventFileList = dir([DB_path{l},'/*.wav']);
        rng('default');
        ix = randperm(length(EventFileList));
        EventFileList = EventFileList(ix);
        s_full = zeros(p.train_seq_len_max,1);
        s_len_cnt = 0;
        batch_cnt = 0;
        file_num = p.load_file_num;
        if file_num > length(EventFileList) || file_num == 0
            file_num = length(EventFileList);
        end
        
        for i=1:file_num
            [filename, ext] = strtok(EventFileList(i).name, '.');
            disp(['Training:', filename]);
            event = [DB_path{l},'/',filename, ext];
            [s, fs_s] = wavread(event);
            
            %Normalize sampling frequency and channel to system default
            s = mean(s,2); %Downmix to mono channel
            
            if fs_s < 8000
                continue;
            else
                s = resample(s, p.fs, fs_s);
            end
            
            %Try load YAML metadata and reform sub-wav sequence with valid regions
            meta = [DB_path{l},'/',filename, '.yaml'];
            meta_exist = fopen(meta);
            if meta_exist ~= -1 
                meta_struct = YAML.read(meta);
                v = meta_struct.valid_segments;
                
                s_tmp = 0;
                for k = 1 : size(v,1)
                    smpl_start = round(p.fs * v(k,1)); 
                    smpl_end = round(p.fs * v(k,2));
                    
                    %In case annotation has bug
                    if smpl_start == 0
                        smpl_start = 1;
                    end
                    
                    if smpl_end > length(s)
                        smpl_end = length(s);
                    end
                    
                    if s_tmp == 0
                       s_tmp = [s_sil; s(smpl_start:smpl_end, 1)];
                    else
                       s_tmp = [s_tmp; s(smpl_start:smpl_end, 1)];
                    end
                end
                s = s_tmp;
            else
                if VAD_set(l)
                    %Perform simple VAD to exclude silence part
                    p.bg_len = 0.05 * p.fs; %50ms
                    p.min_voiced_len = 0.5 * p.fs; %0.3s
                    p.min_unvoiced_len = 0.4 * p.fs; %0.3s
                    p.thr = 0.7;
                    vad = vadenergy_simple(s, p);
                    s = nonzeros(s.*vad);
                end
                s = [s_sil; s];
            end
            
            %Normalize sub-wav clip
            s = s ./ sqrt(var(s));
            s = s ./ max(abs(s)) .* 30000;
            s_len = length(s);
            if s_len >  p.train_seq_len_max
               s = s(1:p.train_seq_len_max); 
               s_len = length(s);
            end
            s_full(1 + s_len_cnt : s_len + s_len_cnt) = s;
            
            s_len_cnt = s_len_cnt + s_len;
            if s_len_cnt >= p.train_seq_len_max || i == file_num
                s_full = [s_full(1:p.train_seq_len_max, 1); s_sil];
                if s_len_cnt <= p.train_seq_len_max
                    s_full = [s_full(1:s_len_cnt, 1); s_sil];
                end
                s_full = s_full + 1;
                    
              %% Extract feature for NMF
                %Feature1: DFT Magnitude
                [TF_mag, ~] = stft_fft(s_full, p.framelength, p.frameshift, p.fftlength, DC_bin_set(l), p.win_STFT, p.preemph);
                TF_mag = TF_mag(:,any(TF_mag,1)); %Exclude all-zero column
                [TF_mag, ~] = frame_splice(TF_mag,p);
                TF_mag = TF_mag .^ p.pow + p.nonzerofloor;

                %Feature2: Mel Magnitude
                m = size(TF_mag,2);
                n = p.fftlength/2 + 1;
                melmat = mel_matrix(p.fs, p.F_order, p.fftlength, 1, p.fs/2)'; %Get Mel Matrix
                TF_Mel = zeros(p.F_order*(p.Splice * 2 + 1),m);
                for k = 1 : p.Splice * 2 + 1
                    TF_Mel(1+(k-1)*p.F_order : k*p.F_order, :) = ...
                        melmat*TF_mag(1+(k-1)*n : k*n, :);
                end
                
              %% Train NMF Parameters
                if p.train_Exemplar == 0
                    p.w_update_ind = true(p.cluster_buff*R,1);
                    p.h_update_ind = true(p.cluster_buff*R,1);
                    if batch_cnt > 0
                        p.init_w = B_DFT_init; %Given from Exemplar basis as initialization
                    else
                        p.r = R;
                    end
                    [B_DFT_init, A_DFT_init] = sparse_nmf(TF_mag, p);
                    if batch_cnt > 0
                        A_DFT_tot = [A_DFT_tot, A_DFT_init];
                    else
                        A_DFT_tot = A_DFT_init;
                    end
                    
                    if batch_cnt > 0
                        p.init_w = B_Mel_init; %Given from Exemplar basis as initialization
                    else
                        p.r = R;
                    end
                    [B_Mel_init, A_Mel_init] = sparse_nmf(TF_Mel, p);
                    if batch_cnt > 0
                        A_Mel_tot = [A_Mel_tot, A_Mel_init];
                    else
                        A_Mel_tot = A_Mel_init;
                    end
                else
                    rng('default'); rng(1);
                    sample_idx =  randsample(size(TF_mag,2),p.cluster_buff*R);
                    B_DFT_init = TF_mag(:,sample_idx);
                    B_Mel_init = TF_Mel(:,sample_idx);
                    A_DFT_init = 0;
                    A_Mel_init = 0;
                end
                batch_cnt = batch_cnt + 1;
                wavwrite(s_full./32767, p.fs, [dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_train_batch',num2str(batch_cnt),'.wav']);
                s_full = zeros(p.train_seq_len_max,1);
                s_len_cnt = 0;
                
            end
        end
        clear('s_full');
                
        % Basis Normalization
        wn = sqrt(sum(B_DFT_init.^2));
        B_DFT_init  = bsxfun(@rdivide,B_DFT_init,wn) + 1e-9;
        wn = sqrt(sum(B_Mel_init.^2));
        B_Mel_init  = bsxfun(@rdivide,B_Mel_init,wn) + 1e-9;
    
        B_DFT_sub = B_DFT_init;
        B_Mel_sub = B_Mel_init;
        A_DFT_sub = A_DFT_tot;
        A_Mel_sub = A_Mel_tot;
        
        save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Basis.mat'],'B_DFT_sub', 'B_Mel_sub', 'p', '-v7.3');
        save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Activation.mat'],'A_DFT_sub', 'A_Mel_sub', 'p', '-v7.3');
    else
        p_org = p;
        load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Basis.mat']);
        load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Activation.mat']);

        p = p_org; 
    end
    
   %% Clear event-wise buffer
    try
        p = rmfield(p, {'w_update_ind', 'h_update_ind', 'init_w', 'init_h'});
    catch
    end
    
    if p.train_MLD == 1
        A_DFT_tot = 0;
        A_Mel_tot = 0;
        fexist = fopen([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Basis_MLD.mat']);
        if fexist == -1 || p.ForceRetrain_MLD
            disp(['------',num2str(l),'-th event train (MLD)------']);
            
            R_g = round(R / p.R_c);
            %% Obtain centroid of each spectral cone using K-means clustering
            [~, C_DFT, ~, ~]= kmeans(B_DFT_sub', R_g, ...
                'distance', 'cityblock', ...
                'emptyaction', 'singleto', ...
                'onlinephase', 'off', ...
                'start', 'sample');
            C_DFT = C_DFT';
            C_DFT = kron(C_DFT, ones(1, p.R_c)); %Span to match size with W
            
            [~, C_Mel, ~, ~]= kmeans(B_Mel_sub', R_g, ...
                'distance', 'cityblock', ...
                'emptyaction', 'singleto', ...
                'onlinephase', 'off', ...
                'start', 'sample');
            C_Mel = C_Mel';
            C_Mel = kron(C_Mel, ones(1, p.R_c)); %Span to match size with W
            
           %% Retrain batch spectrograms with SNMF_MLD
            BatchFileList = dir([dir_Basis,'/',B_name_full{l},'/*.wav']);
            for batch_cnt = 1:length(BatchFileList)
                
                s_full = wavread([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_train_batch',num2str(batch_cnt),'.wav']); 
            
                %Feature1: DFT Magnitude
                [TF_mag, ~] = stft_fft(s_full, p.framelength, p.frameshift, p.fftlength, DC_bin_set(l), p.win_STFT, p.preemph);
                TF_mag = TF_mag(:,any(TF_mag,1)); %Exclude all-zero column
                [TF_mag, ~] = frame_splice(TF_mag,p);
                TF_mag = TF_mag .^ p.pow + p.nonzerofloor;

                %Feature2: Mel Magnitude
                m = size(TF_mag,2);
                n = p.fftlength/2 + 1;
                melmat = mel_matrix(p.fs, p.F_order, p.fftlength, 1, p.fs/2)'; %Get Mel Matrix
                %     melmat_splice = repmat(melmat,2*p.Splice+1,2*p.Splice+1);
                TF_Mel = zeros(p.F_order*(p.Splice * 2 + 1),m);
                for k = 1 : p.Splice * 2 + 1
                    TF_Mel(1+(k-1)*p.F_order : k*p.F_order, :) = ...
                        melmat*TF_mag(1+(k-1)*n : k*n, :);
                end
                
                p.w_update_ind = true(p.cluster_buff*R,1);
                p.h_update_ind = true(p.cluster_buff*R,1);
                if batch_cnt > 1
                    p.init_w = B_DFT_MLD; %Given from Exemplar basis as initialization
                else
                    p.r = R;
                end
                [B_DFT_MLD, A_DFT_MLD] = sparse_nmf_MLD(TF_mag, C_DFT, p);
                if batch_cnt > 1
                    A_DFT_tot = [A_DFT_tot, A_DFT_MLD];
                else
                    A_DFT_tot = A_DFT_MLD;
                end
                
                if batch_cnt > 1
                    p.init_w = B_Mel_MLD; %Given from Exemplar basis as initialization
                else
                    p.r = R;
                end
                [B_Mel_MLD, A_Mel_MLD] = sparse_nmf_MLD(TF_Mel, C_Mel, p);
                if batch_cnt > 1
                    A_Mel_tot = [A_Mel_tot, A_Mel_MLD];
                else
                    A_Mel_tot = A_Mel_MLD;
                end
            end
            
            % Basis Normalization
            wn = sqrt(sum(B_DFT_MLD.^2));
            B_DFT_MLD  = bsxfun(@rdivide,B_DFT_MLD,wn) + 1e-9;
            wn = sqrt(sum(B_Mel_MLD.^2));
            B_Mel_MLD  = bsxfun(@rdivide,B_Mel_MLD,wn) + 1e-9;
            
            B_DFT_sub = B_DFT_MLD;
            B_Mel_sub = B_Mel_MLD;
            A_DFT_sub = A_DFT_tot;
            A_Mel_sub = A_Mel_tot;
            save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Basis_MLD.mat'],'B_DFT_sub', 'B_Mel_sub', 'p', '-v7.3');
            save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Activation_MLD.mat'],'A_DFT_sub', 'A_Mel_sub', 'p', '-v7.3');
        else
            p_org = p;
            load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Basis_MLD.mat']);
            load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_Activation_MLD.mat']);
            
            p = p_org;
        end
    end
    
    if p.train_HMM_NMF
        fexist = fopen([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_HMM.mat']);
        if fexist == -1 || p.ForceRetrain_MLD
            disp(['------',num2str(l),'-th event train (HMM)------']);
            [~, dict_seq] = max(A_Mel_sub); dict_seq = round(dict_seq * 0.5);
            idx_sil = mean(B_Mel_sub * A_Mel_sub) <= 0.00001 .* mean(mean(B_Mel_sub * A_Mel_sub));
            dict_seq(idx_sil) = p.R_Dict + 1; %Set Silence observation
            [prior, transmat, emismat] = init_HMM(dict_seq, p.Q, p.R_Dict + 1);
            
             save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_HMM.mat'],'prior', 'transmat', 'emismat', '-v7.3');
        else
             load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_HMM.mat']);
        end
    end
end

p_out = p;
clear('B_DFT_sub', 'B_Mel_sub', 'A_DFT_tot', 'A_Mel_tot');
fclose('all');
end