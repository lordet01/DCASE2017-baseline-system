function [loglik, gamma]=HMM_loglik(Data, p_start, A, phi, p)

[M,Q] = size(phi.B);

[seq_len, feat_num] = size(Data);

% % use model to compute log likelihood
logp_xn_given_zn = Gmm_logp_xn_given_zn(Data, phi);
[gamma,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);

% % mu_t = zeros(feat_num,M,seq_len);
% % Sigma_t = zeros(feat_num,feat_num,M,seq_len);
% % prior_t = phi.B * exp(gamma)';
% % for k = 1:feat_num
% %    mu_t(k,:,:) = squeeze(phi.mu(k,:,:)) * exp(gamma)';
% %    Sigma_t(k,k,:,:) = squeeze(phi.Sigma(k,k,:,:)) * exp(gamma)';
% % end
% % 
% % mu_out = zeros(feat_num,seq_len);
% % for t = 1:seq_len
% %     mu_out(:,t) = squeeze(mu_t(:,:,t)) * prior_t(:,t);
% % %     mu_out(:,t) = sum(squeeze(mu_t(:,:,t)),2);
% % %     pdf_seq = Gmmpdf(data_test(1:t,:), prior_t(:,t), mu_t(:,:,t), Sigma_t(:,:,:,t));
% % end
% % 
% % % datum = gaussPDF(Data, squeeze(phi.mu(k,:,:)) * exp(gamma)', Sigma)
% % figure();
% % mesh(exp(Data-1)');
% % figure();
% % mesh(exp(mu_out-1));
% % mu_out = mu_out;