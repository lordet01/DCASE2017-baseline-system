function run_dictionary_train(dir_Basis, DB_path, DC_freq_set, VAD_set, event_num, B_name_full, feat_type, p)

addpath('src');
addpath(genpath('toolbox/yaml')); 
addpath(genpath('toolbox/LFCC_rastamat')); 

DC_bin_set = floor(DC_freq_set ./ (p.fs / p.fftlength) + 0.5);
s_sil = zeros(round(p.fs * p.sil_len), 1);

%% Get Mel, Linear Transform Matrix
if strcmp(feat_type, 'mel')
    featmat = fft2melmx(p.fftlength, p.fs, p.F_order, 1, p.DCfreq, p.fhigh, 1, 1);
    featmat = featmat(:,1:round(p.fftlength*0.5)+1);
elseif strcmp(feat_type, 'lin')
    params.brkfrq = p.fs/2; % to get linear scale
    params.dorasta = 0 ;
    featmat = fft2melmxXinhui(p.fftlength, p.fs, p.F_order, 1, p.DCfreq, p.fhigh, params, 0, 1);
    featmat = featmat(:,1:round(p.fftlength*0.5)+1);
end

A_DFT = 0;
A_Mel = 0;
for l = 1:event_num
    fexist = fopen([dir_Basis, '/',B_name_full{l},'/', B_name_full{l},'_',feat_type,'_Dict.mat']);
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
        
        TF_Patch_Mel = 0;
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
                TF_Mel = zeros(p.F_order*(p.Splice * 2 + 1),m);
                for k = 1 : p.Splice * 2 + 1
                    TF_Mel(1+(k-1)*p.F_order : k*p.F_order, :) = ...
                        featmat*TF_mag(1+(k-1)*n : k*n, :);
                end
                TF_Lin = zeros(p.F_order*(p.Splice * 2 + 1),m);
                for k = 1 : p.Splice * 2 + 1
                    TF_Lin(1+(k-1)*p.F_order : k*p.F_order, :) = ...
                        featmat*TF_mag(1+(k-1)*n : k*n, :);
                end
                
                %Form a set of spectral patch
                T_patch = floor(size(TF_Mel,2) / p.Patch_Len) * p.Patch_Len;
                for ii = 1 : p.Patch_shift : T_patch-p.Patch_Len+1
                    TF_Patch_new = zeros(1, p.Patch_Len, size(TF_Mel,1));
                    TF_Patch_new(1,:,:) = TF_Mel(:,ii:ii+p.Patch_Len-1)';
                    if TF_Patch_Mel == 0
                        TF_Patch_Mel = TF_Patch_new;
                    else
                        TF_Patch_Mel = cat(1, TF_Patch_Mel, TF_Patch_new);
                    end
                end
                
                batch_cnt = batch_cnt + 1;
                wavwrite(s_full./32767, p.fs, [dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_train_batch',num2str(batch_cnt),'.wav']);
                s_full = zeros(p.train_seq_len_max,1);
                s_len_cnt = 0;

            end
        end
        clear('s_full');
        
        %% K-means clustering of spectral patch
        TF_Patch_Mel2d = reshape(TF_Patch_Mel, [size(TF_Patch_Mel,1), size(TF_Patch_Mel,2) * size(TF_Patch_Mel,3)]);
        TF_Patch_Mel2d = log(TF_Patch_Mel2d+1); %Apply log for better clustering 
        [IDX, ~, ~, ~]= kmeans(TF_Patch_Mel2d, p.R_Dict, ...
            'Distance', 'cityblock', ...
            'EmptyAction', 'singleto', ...
            'OnlinePhase', 'off', ...
            'Start', 'sample');
        
        %% Regroup spectral patch by cluter IDX
        G_data = {1, p.R_Dict};
        B_g = {1, p.R_Dict};
        LAMBDA_g = {1, p.R_Dict};
        B_event = 0;
        for ig = 1:p.R_Dict
            G_data{ig} = TF_Patch_Mel(IDX==ig,:,:);
            G_data_num = size(G_data{ig},1);
            
            %% Train Group-wise basis
            p.r = p.R_c;
            TF_g = reshape(G_data{ig}, [G_data_num * p.Patch_Len, p.F_order]);
            p.h_weight = 1; %Don;t Use in Initial Training
            [B_g{ig}, ~] = sparse_nmf(TF_g', p);
            
            %% Obtain Activation sequence for a given basis
            TF_g_sub = G_data{ig};
            A_g = {1, G_data_num};
            for jg = 1:G_data_num
                TF_g_slice = squeeze(TF_g_sub(jg,:,:));
                p.w_update_ind = false(p.R_c,1);
                p.r = p.R_c;
                p.init_w = B_g{ig};
                [~, A_g{jg}] = sparse_nmf(TF_g_slice', p);
                A_g{jg} = log(A_g{jg}'+1); %preprocessing prior to HMM train
            end
            
            if G_data_num < p.M
                for jg = G_data_num+1:p.M
                    A_g{jg} = A_g{jg -  G_data_num};
                end
            end
            
            %% Train HMM parameters for a given activation sequence
            Q_plus = floor(G_data_num / 1000);
             [lambda.p_start, lambda.A, lambda.phi, ~] = train_mHMM(A_g, p.Q+Q_plus, p.M, p);
            LAMBDA_g{ig} = lambda;
% %             test_mHMM(A_g{2}, lambda.p_start, lambda.A, lambda.phi, p);
            
            %Clear event-wise buffer
            try
                p = rmfield(p, {'w_update_ind', 'init_w'});
            catch
            end
            
            if B_event == 0
                B_event = B_g{ig};
            else
                B_event = [B_event, B_g{ig}];
            end
        end
        
% %         % Basis Normalization
% %         wn = sqrt(sum(B_DFT_init.^2));
% %         B_DFT_init  = bsxfun(@rdivide,B_DFT_init,wn) + 1e-9;
% %         wn = sqrt(sum(B_Mel_init.^2));
% %         B_Mel_init  = bsxfun(@rdivide,B_Mel_init,wn) + 1e-9;

        save([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_',feat_type,'_Dict.mat'],'B_event', 'LAMBDA_g', 'p', '-v7.3');
    else
        p_org = p;
        load([dir_Basis,'/',B_name_full{l},'/', B_name_full{l},'_',feat_type,'_Dict.mat']);
        p = p_org;
    end
    
end

%Clear garbage buffers regarding NMF train
try
    p = rmfield(p, {'w_update_ind', 'init_w'});
catch
end

clear('B_event');
fclose('all');
end