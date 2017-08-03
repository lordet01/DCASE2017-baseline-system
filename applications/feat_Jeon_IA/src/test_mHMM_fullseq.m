function test_mHMM_fullseq(x, seq_len, p_start, A, phi, p)

[M,Q] = size(phi.B);

 %% Refine training sequence into sub-size
Data = log(x + 1)'; %Data preprocessing
feat_num = size(Data,2);
seq_len = floor(seq_len ./ p.hoptime);
data_num = floor(size(Data,1) / seq_len);
Data = Data(1:data_num * seq_len, :);
Data = reshape(Data, [data_num, seq_len, size(Data,2)]);
rng('default');
ix = randperm(size(Data,1));


% % use model to compute log likelihood
data_test = reshape(Data, [data_num*seq_len, feat_num]);
logp_xn_given_zn = Gmm_logp_xn_given_zn(data_test, phi);
[gamma,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);

mu_t = zeros(feat_num,M,data_num*seq_len);
Sigma_t = zeros(feat_num,feat_num,M,data_num*seq_len);
prior_t = phi.B * exp(gamma)';
for k = 1:feat_num
   mu_t(k,:,:) = squeeze(phi.mu(k,:,:)) * exp(gamma)';
   Sigma_t(k,k,:,:) = squeeze(phi.Sigma(k,k,:,:)) * exp(gamma)';
end

mu_out = zeros(feat_num,data_num*seq_len);
for t = 1:data_num*seq_len
    mu_out(:,t) = squeeze(mu_t(:,:,t)) * prior_t(:,t);
%     mu_out(:,t) = sum(squeeze(mu_t(:,:,t)),2);
%     pdf_seq = Gmmpdf(data_test(1:t,:), prior_t(:,t), mu_t(:,:,t), Sigma_t(:,:,:,t));
end

% datum = gaussPDF(Data, squeeze(phi.mu(k,:,:)) * exp(gamma)', Sigma)
figure();
mesh(exp(data_test-1)');
figure();
mesh(exp(mu_out-1));
mu_out = mu_out;