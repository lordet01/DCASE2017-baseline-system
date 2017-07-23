O = 5;
Q = 3;

%% For left to right HMM
left_to_right_model = 1;
if left_to_right_model == 1
    T = rand(Q,Q);
    size_mat = size(T);
    for i = 1 : size_mat(1,1);
        for j = 1: size_mat(1,2);
            if (j < i)
                T(i,j) = 0;
            end
        end
    end
    transmat0=mk_stochastic(T);
else
    transmat0=mk_stochastic(rand(Q,Q));
end
prior0=normalise(rand(Q,1));
obsmat0 = mk_stochastic(rand(Q,O));


% training data
T = 10000;
nex = 1;
data = dhmm_sample(prior0, transmat0, obsmat0, T, nex);
data = data';

% % initial guess of parameters
% prior1 = normalise(rand(Q,1));
% transmat1 = mk_stochastic(rand(Q,Q));
% obsmat1 = mk_stochastic(rand(Q,O));

prior = prior0;
transmat = transmat0;
obsmat = obsmat0;
% improve guess of parameters using EM
iter = 30;
for i = 1 : iter
    [LL, prior, transmat, obsmat] = dhmm_em(data, prior, transmat, obsmat, 'max_iter', 1);
end
prior2 = prior;
transmat2 = transmat;
obsmat2 = obsmat;

% use model to compute log likelihood
[loglik, ~, alpha, beta] = dhmm_logprob(data, prior2, transmat2, obsmat2);
% log lik is slightly different than LL(end), since it is computed after the final M step
