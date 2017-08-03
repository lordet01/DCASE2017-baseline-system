function [x_hat_i,d_hat_i, x_tilde_i, feat, g] = nmf_sep_event_RT_DCASE17_Dict(y, l, g, p)

%% Global buffer initialize
Ym = g.Ym;
Yp = g.Yp;
Xm_hat = g.Xm_hat;
Dm_hat = g.Dm_hat;
Xm_Mel = g.Xm_Mel;
Dm_Mel = g.Dm_Mel;
x_hat = g.x_hat;
d_hat = g.d_hat;
x_tilde = g.x_tilde;
blk_cnt = g.blk_cnt;
B_Mel_x = g.B_Mel_x;
B_Mel_d = g.B_Mel_d;
B_Lin_x = g.B_Lin_x;
B_Lin_d = g.B_Lin_d;
HMM_Mel_x = g.HMM_Mel_x;
HMM_Mel_d = g.HMM_Mel_d;
HMM_Lin_x = g.HMM_Lin_x;
HMM_Lin_d = g.HMM_Lin_d; 
Ad_blk = g.Ad_blk;
A_d = g.A_d;
A_buff = g.A_buff;
lambda_d_blk = g.lambda_d_blk;
lambda_dav = g.lambda_dav;
lambda_Gy = g.lambda_Gy;
l_mod_lswitch = g.l_mod_lswitch;
Xm_tilde = g.Xm_tilde;
r_blk = g.r_blk;
Xm_sep_1d = g.Xm_sep_1d;
Dm_sep_1d = g.Dm_sep_1d;
Xm_tilde_last = g.Xm_tilde_last;
A_w_Mel_1d = g.A_w_Mel_1d;

%% set local parameters
[n1,~] = size(B_Mel_d);
[~,R_d] = size(B_Lin_d);
[n2,R_x] = size(B_Lin_x);
G_x = floor(R_x / p.R_c);
G_d = floor(R_d / p.R_c);
if p.adapt_train_N == 1
    G_d = G_d - floor(p.R_a / p.R_c);
end
n1_unit = floor(n1 /  (2*p.Splice+1));
n2_unit = floor(n2 /  (2*p.Splice+1));
if strcmp(p.B_sep_mode, 'Mel')
    nf_unit = n1_unit;
else
    nf_unit = n2_unit;
end
r = R_x+R_d;
m = p.blk_len_sep;
h = p.blk_hop_sep;
sz = p.framelength;
fftlen = p.fftlength;
fftlen2 = round(fftlen/2+1);
fftlen2_splice = fftlen2 * (2*p.Splice+1);
splice = p.Splice;
if blk_cnt > h
    blk_cnt = mod(blk_cnt,h);
end

%% STFT
y = filter([1 -p.preemph], 1, y);
y = p.win_STFT .* y';
y_pad = [y; zeros(fftlen-sz,1)];
Y = fft(y_pad);
Yp_t = angle(Y(1:fftlen2));
Ym_t = abs(Y(1:fftlen2)) .^ p.pow;

%Set zero for LPF effect
Ym(1:p.DCbin) = zeros(p.DCbin,1);

%FIt zero to floor value
Ym(:,1) = Ym(:,1) + p.nonzerofloor;

%Block shift
if m > 1
    Ym(:, 1:m-1) = Ym(:, 2:m); %Block shift
    Yp(:, 1:m-1) = Yp(:, 2:m);
end

%Block update
if m > 1
    Ym(1:fftlen2_splice-fftlen2,m) = Ym(1+fftlen2:fftlen2_splice,m-1);
    Yp(1:fftlen2_splice-fftlen2,m) = Yp(1+fftlen2:fftlen2_splice,m-1);
else
    Ym(1:fftlen2_splice-fftlen2,m) = Ym(1+fftlen2:fftlen2_splice,m);
    Yp(1:fftlen2_splice-fftlen2,m) = Yp(1+fftlen2:fftlen2_splice,m);
end
Ym(fftlen2_splice-fftlen2+1:fftlen2_splice,m) = Ym_t;
Yp(fftlen2_splice-fftlen2+1:fftlen2_splice,m) = Yp_t;

%Splice processing range after the separation
splice_ext_Mel = (1+splice*n1_unit):(splice+1)*n1_unit;
splice_ext = (1+splice*fftlen2):(splice+1)*fftlen2;
if blk_cnt==h
    
    %% Feature Frequency Scale Conversion (DFT to Mel)
    if strcmp(p.B_sep_mode, 'Mel')
        Ym_Mel = zeros(n1, m);
        for k = 1 : p.Splice * 2 + 1
            Ym_Mel(1+(k-1)*n1_unit : k*n1_unit, :) = ...
                g.melmat*shiftdim(Ym(1+(k-1)*fftlen2 : k*fftlen2, :));
        end
        Y_sep = Ym_Mel;
    elseif strcmp(p.B_sep_mode, 'Lin')
        Ym_Lin = zeros(n2, m);
        for k = 1 : p.Splice * 2 + 1
            Ym_Lin(1+(k-1)*n1_unit : k*n1_unit, :) = ...
                g.linmat*shiftdim(Ym(1+(k-1)*fftlen2 : k*fftlen2, :));
        end
        Y_sep = Ym_Lin;
    elseif strcmp(p.B_sep_mode, 'DFT')
        Y_sep = Ym;
    end
    
    %         %Power normaliation of mel_mag_blk matched to Ym
%         vn = sqrt(sum(Ym_Mel.^2));
%         tn = sqrt(sum(Ym.^2));
%         Ym_Mel  = bsxfun(@rdivide,Ym_Mel,vn) + 1e-9;
%         Ym_Mel = bsxfun(@times,Ym_Mel,tn);

if p.ProcBypass == 0
    %% 1) Perform SNMF separation
    if p.basis_update_N
        p.w_update_ind = [false(R_x + R_d - p.R_semi,1); true(p.R_semi,1)]; % Semi-supervised:
        %         B_Mel = [B_Mel_x, rand(n1,R_d)];
        %         B_Lin = [B_Lin_x, rand(n2,R_d)];
    elseif p.basis_update_E
        p.w_update_ind = [true(p.R_semi,1); false(R_x + R_d - p.R_semi,1)]; % Semi-supervised:
        %         B_Mel = [rand(n1,R_x), B_Mel_d];
        %         B_Lin = [rand(n2,R_x), B_Lin_d];
    elseif p.basis_update_N && p.basis_update_E
        p.w_update_ind = true(r,1);
        %         B_Mel = rand(n1,R_x+R_d);
        %         B_Lin = rand(n2,R_x+R_d);
    else
        p.w_update_ind = false(r,1); % Supervised
    end
    B_Mel = [B_Mel_x, B_Mel_d];
    B_Lin = [B_Lin_x, B_Lin_d];
    
    if strcmp(p.B_sep_mode, 'Mel')
        p.init_w = B_Mel; %initialization
    elseif strcmp(p.B_sep_mode, 'Lin')
        p.init_w = B_Lin; %initialization
    end
    
    p.h_update_ind = true(r,1);
    
    %% Check Sparsity between Input and Basis
    if p.SparseCheck == 1
        Y_sep_t = Ym(splice_ext_Mel, :);
        B_t = p.init_w(splice_ext_Mel, :);
        n_tmp = size(Y_sep_t,1);
        SC_BandL = floor(p.SC_RatioL * n_tmp);
        SC_BandH = floor(p.SC_RatioH * n_tmp);
        [Q_Y]= blk_sparse_single(Y_sep_t(SC_BandL:SC_BandH, :), p);
        [Q_B]= blk_sparse_single(B_t(SC_BandL:SC_BandH, :), p);
        
        Q_H = zeros(r, m);
        for t = 1:m
            Q_H(:,t) = 1 - abs(Q_B' - Q_Y(t));
        end
        simil_scale = Q_H .^ p.SC_pow;
        p.h_weight = simil_scale;
    end
    
    %Sparse NMF-based Source Separation
    [~, A] = sparse_nmf(Y_sep, p);
    
    
    %% Measure likelihood of each dictionary
    if p.TransitionCheck == 1
        %Update Activation buffer
        if p.blk_len_sep < p.Patch_Len
            A_buff(:, 1:p.Patch_Len-p.blk_len_sep) = A_buff(:, p.blk_len_sep+1:p.Patch_Len);
            A_buff(:, p.Patch_Len-p.blk_len_sep+1:p.Patch_Len) = A;
        else
            A_buff = A;
        end

        loglik_x_Mel = zeros(G_x, 1);
        for ig = 1:G_x
            A_dict = A_buff((ig-1)*p.R_c+1:ig*p.R_c,:)';
            A_dict = log(A_dict+1);
            p_start = HMM_Mel_x{ig}.p_start; trans = HMM_Mel_x{ig}.A; phi = HMM_Mel_x{ig}.phi;
            if nnz(A_dict) == 0
                loglik_x_Mel(ig) = -100000;
            else
                [loglik_x_Mel(ig), ~] = HMM_loglik(A_dict, p_start, trans, phi, p);
            end
        end


        %calculate activation weights
        w_Mel = 1./(loglik_x_Mel.^p.TC_pow);
        A_w_Mel = w_Mel ./ sum(w_Mel);
        mu_A_w_Mel = mean(A_w_Mel);
        g.A_w_Mel_1d = A_w_Mel;
        A_interp = (A_w_Mel - A_w_Mel_1d) ./ (p.Patch_Len-1);
        A_w_x_Mel = zeros(G_x, p.Patch_Len);
        for ig = 1:G_x
            if A_interp(ig) == 0
                A_w_x_Mel(ig,:) = repmat(A_w_Mel(ig), [1, p.Patch_Len]);
            else
                A_w_x_Mel(ig,:) =  (A_w_Mel_1d(ig) : A_interp(ig) : A_w_Mel(ig));
            end
        end
        A_w_x_Mel = kron(A_w_x_Mel, ones(p.R_c,1));

        A_w_Mel = [A_w_x_Mel; ones(R_d, p.blk_len_sep) .* mu_A_w_Mel];
        A = A .* A_w_Mel;
    end

    %Multiclass separation (Event)
    for i = 1:p.EVENT_NUM
        if i == p.EVENT_NUM
            R_x_i = p.EVENT_RANK(i):R_x;
        else
            R_x_i = p.EVENT_RANK(i):p.EVENT_RANK(i+1)-1;
        end
        if strcmp(p.B_sep_mode, 'Mel')
            tmp_Xm_hat_Mel = B_Mel(:,R_x_i)*A(R_x_i, :);
            
            %Get Last frame from supervector
            Xm_Mel(i,:,:) = tmp_Xm_hat_Mel(:,:);
        else
            tmp_Xm_hat = B_Lin(:,R_x_i)*A(R_x_i, :);
            Xm_hat(i,:,:) = tmp_Xm_hat(:,:);
        end
    end
    
    %Multiclass separation (Noise)
    for i = 1:p.NOISE_NUM
        if i == p.NOISE_NUM
            R_d_i = R_x+p.NOISE_RANK(i):R_x+R_d;
        else
            R_d_i = R_x+p.NOISE_RANK(i):R_x+p.NOISE_RANK(i+1)-1;
        end
        
        if strcmp(p.B_sep_mode, 'Mel')
            tmp_Dm_hat_Mel = B_Mel(:,R_d_i)*A(R_d_i, :);
            
            %Get Last frame from supervector
            Dm_Mel(i,:,:) = tmp_Dm_hat_Mel;
        else
            tmp_Dm_hat = B_Lin(:,R_d_i)*A(R_d_i, :);
            Dm_hat(i,:,:) = tmp_Dm_hat(:,:);
        end
    end
    if strcmp(p.B_sep_mode, 'Mel')
        Xm_Mel_sum = shiftdim(sum(Xm_Mel,1));
        Dm_Mel_sum = shiftdim(sum(Dm_Mel,1));
        Xm_sep = Xm_Mel_sum;
        Dm_sep = Dm_Mel_sum;
    else
        Xm_hat_sum = shiftdim(sum(Xm_hat,1));
        Dm_hat_sum = shiftdim(sum(Dm_hat,1));
        Xm_sep = Xm_hat_sum;
        Dm_sep = Dm_hat_sum;
    end
    
    %% Calculate Block Sparsity
    if p.blk_sparse
        [Q, r_blk]= blk_sparse(Xm_sep, Dm_sep, r_blk, l, p);
    else
        Q = ones(n1, m) .* 0.5;
    end
    
    %% 3) Enhancement Filter Construction
    if strcmp(p.ENHANCE_METHOD, 'Wiener') || strcmp(p.ENHANCE_METHOD, 'MMSE')
        %Estimate smoothed noise PSD
        if l == 1
            lambda_dav=Y_sep;
        end
        
        %% A ratio calculation %20160315
        A_d_mag = (sum(A(1+R_x:R_x+R_d,m)) / R_d);
        A_x_mag = (sum(A(1:R_x,m)) / R_x);
        beta = 20*log10(A_d_mag / (A_x_mag));
        beta = beta * p.G_beta;
        
        if beta < p.G_beta
            beta = p.G_beta;
        elseif beta >= p.G_beta_max
            beta = p.G_beta_max;
        end
        %         disp( beta );
        %         beta = p.G_beta;
        
        lambda_dav=p.alpha_d.*lambda_dav+(1-p.alpha_d).*Dm_sep * beta;
        lambda_d=max(lambda_dav, p.nonzerofloor);  % new version
        
        if strcmp(p.ENHANCE_METHOD, 'Wiener')
            G = max(p.G_floor, Xm_sep) ./ (max(p.G_floor, Xm_sep) + lambda_d * p.G_beta);
        elseif strcmp(p.ENHANCE_METHOD, 'MMSE')
            G = zeros(size(Xm_sep));
            for im = 1:p.blk_len_sep
                if sum(Xm_tilde_last) == 0
                   eta =  (Xm_sep(:,im) .* Q(:,im)) ./ lambda_d(:,im);
                else
                    eta = (p.alpha_eta * Xm_tilde_last + ...
                        (1-p.alpha_eta) * Xm_sep(:,im) .* Q(:,im)) ./ lambda_d(:,im);
                end
                eta = max(p.G_floor, eta);
                G(:,im) =  eta ./ (eta+ones(size(eta)));
                Xm_tilde_last = G(:,im).* Y_sep(:,im);
            end
        end
        G = min(G, 1);
    end
        
    %Noise update Initialization
    if l <= p.init_N_len
        G = zeros(n1, m) + p.G_floor;
    end
    
    Xm_tilde = G.* Y_sep;
    
    %% Extract feature set
    feat = A;
    
    if strcmp(p.B_sep_mode, 'Mel')
        if p.MelOut
            for k = 1 : p.Splice * 2 + 1
                Xm_tilde_out(1+(k-1)*fftlen2 : k*fftlen2, :) = ...
                    g.melmat'*shiftdim(Xm_tilde(1+(k-1)*n1_unit : k*n1_unit, :));
            end
        else
            for k = 1 : p.Splice * 2 + 1
                Xm_tilde_out(1+(k-1)*fftlen2 : k*fftlen2, :) = ...
                    g.melmat'*shiftdim(G(1+(k-1)*n1_unit : k*n1_unit, :));
            end
            Xm_tilde_out = Xm_tilde_out .* Ym;
        end
    elseif strcmp(p.B_sep_mode, 'Lin')
        if p.MelOut
            for k = 1 : p.Splice * 2 + 1
                Xm_tilde_out(1+(k-1)*fftlen2 : k*fftlen2, :) = ...
                    g.linmat'*shiftdim(Xm_tilde(1+(k-1)*n1_unit : k*n1_unit, :));
            end
        else
            for k = 1 : p.Splice * 2 + 1
                Xm_tilde_out(1+(k-1)*fftlen2 : k*fftlen2, :) = ...
                    g.linmat'*shiftdim(G(1+(k-1)*n1_unit : k*n1_unit, :));
            end
            Xm_tilde_out = Xm_tilde_out .* Ym;
        end        
    else
        Xm_tilde_out = G.* Ym;
    end
else
    feat = Y_sep;
    Xm_tilde_out = Ym;
end

    %% --------block-wise inverse STFT-------------
% % %     for i = 1:p.NOISE_NUM
% % %         tmp_Dm_hat = shiftdim(Dm_hat(i,splice_ext,:));
% % %         %         tmp_Dm_hat = Dm_hat_sum;%debug
% % %         d_hat(i,:,:)=synth_ifft_buff(tmp_Dm_hat, Yp(splice_ext,:), sz, fftlen, p.win_ISTFT, p.preemph, p.DCbin_back, p.pow);
% % %         d_hat(i,:,:) = d_hat(i,:,:) * p.overlapscale;
% % %     end
% % %     for i = 1:p.EVENT_NUM
% % %         tmp_Xm_hat = shiftdim(Xm_hat(i,splice_ext,:));
% % %         %         tmp_Xm_hat = Xm_hat_sum;%debug
% % %         x_hat(i,:,:)=synth_ifft_buff(tmp_Xm_hat, Yp(splice_ext,:), sz, fftlen, p.win_ISTFT, p.preemph, p.DCbin_back, p.pow);
% % %         x_hat(i,:,:) = x_hat(i,:,:) * p.overlapscale;
% % %     end
    
    % ----------Phase Compensation---------- %
    if p.phase_comp > 0
        
        %Compensate phase components by SNR (08_K Wojcikci)
        if p.phase_comp == 1
            aprior_SNR = (Dm_hat_sum .^ (1/p.pow) ./ (Xm_tilde_out .^ (1/p.pow) + 1));
            aprior_SNR(1:p.DCbin) = 1;
            aprior_SNR(aprior_SNR<0) = eps;
            Interf = (1-Q) .* (Xm_tilde_out.^ (1/p.pow)) - Q .* (Dm_hat_sum .^ (1/p.pow));
            Interf(Interf<0) = eps;
            D = Dm_hat_sum .^ (1/p.pow) + Interf;
            phase_lambda = aprior_SNR .* D;
        elseif p.phase_comp == 2 % static PSC by K.Wojcicki
            phase_lambda = 0.5 .* p.pc_lambda;
            Interf = 0;
        elseif p.phase_comp == 3 % PSC by K. Paliwal
            phase_lambda = p.pc_alpha  .* (Dm_hat_sum .^ (1/p.pow));
            Interf = 0;
        end
        phase_lambda = [phase_lambda; -1* flipud(phase_lambda(2:end-1))];
        X_tilde_mod = Y + phase_lambda;
        X_tilde_p = angle(X_tilde_mod);
        Xm_tilde_out = (Xm_tilde_out.^ (1/p.pow));
        Xm_tilde_out(Xm_tilde_out<0) = eps;
        x_tilde = synth_ifft_buff(Xm_tilde_out(splice_ext,:), X_tilde_p(:,:), sz, fftlen, p.win_ISTFT, p.preemph, p.DCbin_back, 1);
    else
        x_tilde = synth_ifft_buff(Xm_tilde_out(splice_ext,:), Yp(splice_ext,:), sz, fftlen, p.win_ISTFT, p.preemph, p.DCbin_back, p.pow);
    end
    x_tilde = x_tilde * p.overlapscale;
    
if p.ProcBypass == 0    
    %% 4) Adaptive basis training using enhanced spectrums
    Q_control = (1-mean(Q(:,m))) * p.Ar_up;
    %       disp(Q_control);
    if p.adapt_train_N && (( Q_control*A_d_mag > A_x_mag) || l <= p.init_N_len)
        
        if l <= p.init_N_len %Init condition
            if l < p.Splice*2+1
                D_ref = [repmat(Y_sep(end - l*nf_unit+1 : end - (l-1)*nf_unit), [p.Splice*2+1 - l, 1]); Y_sep(end - l*nf_unit+1 : end)];
            else
                D_ref = Y_sep;
            end
        else
            M_ref =  (1-G);
%             M_ref(1:p.DCbin) = zeros(p.DCbin,1) + p.nonzerofloor;
            D_ref = Y_sep .* M_ref;
        end
        
        %Estimate smoothed noise PSD
        lambda_Gy = D_ref  + 0.0001;
        lambda_d_blk = [lambda_d_blk(:,m+1:p.m_a*m) lambda_Gy];
        %         plot(lambda_d_blk);
        %         contour(lambda_d_blk);
        Ad_blk = [Ad_blk(:,m+1:p.m_a*m) A(R_x+1:R_x+p.R_a,:)];
        
        r_up = Q_control * mean(Ad_blk,2) > A_x_mag;
        r_up_inv = 1 - r_up;
        %           disp(sum(r_up));
        
        Ad_blk_up = Ad_blk .* repmat(r_up, [1 p.m_a*m]);
        Ad_blk_up = Ad_blk_up(any(Ad_blk_up,2),:); %Exclude all-zero rows
        if g.update_switch == floor(p.overlap_m_a * p.m_a)
            
            if strcmp(p.B_sep_mode, 'Mel')
                B_Mel_d_up = (B_Mel_d(:,1:p.R_a) + eps) .* repmat(r_up', [n1 1]);
                B_Mel_d_up = B_Mel_d_up(:,any(B_Mel_d_up,1)); %Exclude all-zero columns
                B_Mel_d_rem = (B_Mel_d(:,1:p.R_a) + eps) .* repmat(r_up_inv', [n1 1]);
                B_Mel_d_rem = B_Mel_d_rem(:,any(B_Mel_d_rem,1)); %Exclude all-zero columns
                
                [~,R_a_up] = size(B_Mel_d_up);
                B_Mel_d_fix = B_Mel_d(:,p.R_a+1 : end);
                if R_a_up > 0
                    p.w_update_ind = true(R_a_up,1);
                    p.h_update_ind = true(R_a_up,1);
                    p.init_w = B_Mel_d_up; %given from exemplar basis as initialization
%                     p.init_h = Ad_blk_up; %given from exemplar basis as initialization
                    [B_d_tmp, ~] = sparse_nmf(lambda_d_blk, p);
                    
                    B_Mel_d = [B_Mel_d_rem, B_d_tmp, B_Mel_d_fix];
%                     mesh(log(B_Mel_d+0.01));
                else
                    B_Mel_d = [B_Mel_d_rem, B_Mel_d_fix];
                end
                
            else
                B_Lin_d_up = B_Lin_d(:,1:p.R_a) .* repmat(r_up', [n1 1]);
                B_Lin_d_up = B_Lin_d_up(:,any(B_Lin_d_up,1)); %Exclude all-zero columns
                B_Lin_d_rem = B_Lin_d(:,1:p.R_a) .* repmat(r_up_inv', [n1 1]);
                B_Lin_d_rem = B_Lin_d_rem(:,any(B_Lin_d_rem,1)); %Exclude all-zero columns
                
                [~,R_a_up] = size(B_Lin_d_up);
                B_Lin_d_fix = B_Lin_d(:,p.R_a+1 : end);
                %                     disp(R_a_up);
                if R_a_up > 0
                    p.w_update_ind = true(R_a_up,1);
                    p.h_update_ind = false(R_a_up,1);
                    p.init_w = B_Lin_d_up; %given from exemplar basis as initialization
                    p.init_h = Ad_blk_up; %given from exemplar basis as initialization
                    [B_d_tmp, ~] = sparse_nmf(lambda_d_blk, p);
                    B_Lin_d = [B_Lin_d_rem, B_d_tmp, B_Lin_d_fix];
                else
                    B_Lin_d = [B_Lin_d_rem, B_Lin_d_fix];
                end
                %                   contour(B_Lin_d);
            end
            
            g.update_switch = 1;
        else
            g.update_switch = g.update_switch + 1;
        end
    end
end
    
%     %Smooth with previous supervector
%     if splice > 0
%         Xm_sep_1d = Xm_sep;
%         Dm_sep_1d = Dm_sep;
%     end
    blk_cnt = 0;
else
    feat = 0;
end
blk_cnt = blk_cnt+1;

%% ----------frame signal writing------------

%Make 2D to 3D matrix for compatibility with NTF functions
tmp = ones(size(x_hat(:,:,blk_cnt),1),1,size(x_hat(:,:,blk_cnt),2));
tmp(:,1,:) = x_hat(:,:,blk_cnt);
x_hat_i = tmp;

tmp_d = ones(1,size(d_hat(:,:,blk_cnt),1),size(d_hat(:,:,blk_cnt),2));
tmp_d(:,1,:) = d_hat(:,:,blk_cnt);
d_hat_i = tmp_d;
x_tilde_i = x_tilde(:,blk_cnt)';

%% Global buffer update
g.Ym = Ym;
g.Yp = Yp;
g.x_hat = x_hat;
g.d_hat = d_hat;
g.x_tilde = x_tilde;
g.blk_cnt = blk_cnt;
g.B_Mel_x = B_Mel_x;
g.B_Mel_d = B_Mel_d;
g.B_Lin_x = B_Lin_x;
g.B_Lin_d = B_Lin_d;
g.A_d = A_d;
g.Ad_blk = Ad_blk;
g.A_buff = A_buff;
g.lambda_d_blk = lambda_d_blk;
g.Xm_sep_1d = Xm_sep_1d;
g.Dm_sep_1d = Dm_sep_1d;

g.lambda_dav = lambda_dav;
g.lambda_Gy = lambda_Gy;
g.l_mod_lswitch = l_mod_lswitch;
g.Xm_tilde = Xm_tilde;
g.r_blk = r_blk;
g.Xm_tilde_last = Xm_tilde(:,p.blk_len_sep);

end