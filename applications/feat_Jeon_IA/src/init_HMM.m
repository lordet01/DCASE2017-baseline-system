function [prior, transmat, emismat]=init_HMM(seq_data, Q, O)


% For left to right HMM
left_to_right_model = 1;
if left_to_right_model == 1
    T = ones(Q,Q);
    size_mat = size(T);
    for i = 1 : size_mat(1,1);
        for j = 1: size_mat(1,2);
            if (j < i) && ~(i == size_mat(1,1) && j == 1)
                T(i,j) = 0;
            end
        end
    end
else
    transmat=mk_stochastic(ones(Q,Q));
end

%Add Non-active state
T = [T, zeros(Q,1)];
T = [T; zeros(1, Q+1)];
T(Q, Q+1) = 1.0;
T(Q+1, 1) = 1.0;
T(Q+1, Q+1) = 1.0;

% % Add Final state
% T = [T, zeros(Q+1,1)];
% T = [T; zeros(1, Q+2)];
% T(Q,Q+2) = 1.0;
% T(Q+1,Q+2) = 1.0;
% 
% % Add Initial state
% T = [zeros(Q+2,1), T];
% T = [zeros(1, Q+3); T];
% T(1,2) = 5.0;
% T(1,Q+2) = 0;


Q = Q + 1;
transmat=mk_stochastic(T);
prior=mk_stochastic(ones(Q,1));
emismat = mk_stochastic(rand(Q,O));
emismat(4,6) = 3.0;
emismat = mk_stochastic(emismat);

% [estTR, estE] = hmmtrain(seq_data,transmat,emismat);
[~, prior, transmat, emismat] = dhmm_em(seq_data, prior, transmat, emismat, 'max_iter', 200, 'thresh', 1e-5);

% % use model to compute log likelihood
% [~, ~, alpha, ~] = dhmm_logprob(seq_data, prior, transmat, emismat);

% B = multinomial_prob(seq_data, emismat);
% path = viterbi_path(prior, transmat, B);
