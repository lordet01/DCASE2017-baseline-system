function [p_start, A, phi, loglik]=train_mHMM_fullseq(x, Q, M, seq_len, num_for_train, p)


%% Define State Transition
% For left to right HMM
left_to_right_model = p.LeftToRight;
if left_to_right_model == 1
    T = ones(Q,Q);
    size_mat = size(T);
    for i = 1 : size_mat(1,1);
        for j = 1: size_mat(1,2);
            if (j < i) %&& ~(i == size_mat(1,1) && j == 1)
                T(i,j) = 0;
            end
        end
    end
else
    T=mk_stochastic(ones(Q,Q));
end

% %Add Non-active state
% T = [T, zeros(Q,1)];
% T = [T; zeros(1, Q+1)];
% T(Q, Q+1) = 1.0;
% T(Q+1, 1) = 1.0;
% T(Q+1, Q+1) = 1.0;

% % Add Final state
% T = [T, zeros(Q,1)];
% T = [T; zeros(1, Q+1)];
% T(Q,Q+1) = 1.0;
% % T(Q+1,Q+2) = 1.0;
% 
% % Add Initial state
% T = [zeros(Q+1,1), T];
% T = [zeros(1, Q+2); T];
% T(1,2) = 1.0;
% % T(1,Q+2) = 0;
% 
% Q = Q + 2;

A0=mk_stochastic(T);
% p_start0 = mk_stochastic(rand(1,Q));
p_start0=[1, zeros(1,Q-1)];

%% Refine training sequence into sub-size
Data = log(x + 1)'; %Data preprocessing
feat_num = size(Data,2);
seq_len = floor(seq_len ./ p.hoptime);
data_num = floor(size(Data,1) / seq_len);
Data = Data(1:data_num * seq_len, :);
Data = reshape(Data, [data_num, seq_len, size(Data,2)]);
rng('default');
ix = randperm(size(Data,1));

if data_num < num_for_train * Q
    num_for_train = floor(data_num / Q);
end

%% Initialize GMM Parameters (Pre-Train)
mu = zeros(feat_num, M, Q);
Sigma = zeros(feat_num, feat_num, M, Q);
for q=1:Q
    Data_GMM_Train = Data(ix((q-1)*num_for_train+1:q*num_for_train),:,:);
    Data_GMM_Train = reshape(Data_GMM_Train, [num_for_train*seq_len, feat_num]);
    [pi, mu_sub, Sigma_sub, loglik] = Gmm(Data_GMM_Train, M, 'cov_type', 'diag', 'cov_thresh', 1e-1, 'restart_num', 1, 'iter_num', 20);
    mu(:,:,q) = mu_sub;
    Sigma(:,:,:,q) = Sigma_sub;
end
B = mk_stochastic(rand(M,Q));
phi0.B = B; phi0.mu = mu; phi0.Sigma = Sigma;

for ii = 1:num_for_train
    Data_HMM_Train{ii} = squeeze(Data(ix(num_for_train+ii),:,:));
end

[p_start, A, phi, loglik] = ChmmGmm(Data_HMM_Train, Q, M, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'cov_type', 'diag', 'cov_thresh', 1e-1, 'iter_num', 20);
disp(['trained models loglike: ', num2str(loglik)]);

