function [w, h, objective] = sparse_nmf_MLD(v, mu, p)

% SPARSE_NMF Sparse NMF with beta-divergence reconstruction error, 
% L1 sparsity constraint, optimization in normalized basis vector space.
%
% [w, h, objective] = sparse_nmf(v, p)
%
% Inputs:
% v:  matrix to be factorized
% p: optional parameters
%     beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
%     cf:       cost function type (default: 'kl'; overrides beta setting)
%               'is': Itakura-Saito divergence
%               'kl': Kullback-Leibler divergence
%               'kl': Euclidean distance
%     sparsity: weight for the L1 sparsity penalty (default: 0)
%     max_iter: maximum number of iterations (default: 100)
%     conv_eps: threshold for early stopping (default: 0, 
%                                             i.e., no early stopping)
%     display:  display evolution of objective function (default: 0)
%     random_seed: set the random seed to the given value 
%                   (default: 1; if equal to 0, seed is not set)
%     init_w:   initial setting for W (default: random; 
%                                      either init_w or r have to be set)
%     r:        # basis functions (default: based on init_w's size;
%                                  either init_w or r have to be set)
%     init_h:   initial setting for H (default: random)
%     w_update_ind: set of dimensions to be updated (default: all)
%     h_update_ind: set of dimensions to be updated (default: all)
%
% Outputs:
% w: matrix of basis functions
% h: matrix of activations
% objective: objective function values throughout the iterations
%
%
%
% References: 
% J. Eggert and E. Korner, "Sparse coding and NMF," 2004
% P. D. O'Grady and B. A. Pearlmutter, "Discovering Speech Phones 
%   Using Convolutive Non-negative Matrix Factorisation
%   with a Sparseness Constraint," 2008
% J. Le Roux, J. R. Hershey, F. Weninger, "Sparse NMF ? half-baked or well 
%   done?," 2015
%
% This implementation follows the derivations in:
% J. Le Roux, J. R. Hershey, F. Weninger, 
% "Sparse NMF ? half-baked or well done?," 
% MERL Technical Report, TR2015-023, March 2015
%
% If you use this code, please cite:
% J. Le Roux, J. R. Hershey, F. Weninger, 
% "Sparse NMF ? half-baked or well done?," 
% MERL Technical Report, TR2015-023, March 2015
%   @TechRep{LeRoux2015mar,
%     author = {{Le Roux}, J. and Hershey, J. R. and Weninger, F.},
%     title = {Sparse {NMF} -?half-baked or well done?},
%     institution = {Mitsubishi Electric Research Labs (MERL)},
%     number = {TR2015-023},
%     address = {Cambridge, MA, USA},
%     month = mar,
%     year = 2015
%   }
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2015 Mitsubishi Electric Research Labs (Jonathan Le Roux,
%                                         Felix Weninger, John R. Hershey)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
str =[];
m = size(v, 1);
n = size(v, 2);


if ~exist('p', 'var')
    p = struct;
end

if ~isfield(p, 'max_iter')
    p.max_iter = 100;
end

if ~isfield(p, 'random_seed')
    p.random_seed = 1;
end

if ~isfield(p, 'sparsity')
    p.sparsity = 5;
end

if ~isfield(p, 'sparsity_eta')
    p.sparsity_eta = 5;
end

if ~isfield(p, 'sparsity_epsilon')
    p.sparsity_epsilon = 1;
end

if ~isfield(p, 'conv_eps')
    p.conv_eps = 0;
end

if ~isfield(p, 'cf')
    p.cf = 'kl';
end

switch p.cf
    case 'is'
        p.beta = 0;
    case 'kl'
        p.beta = 1;
    case 'ed'
        p.beta = 2;
    otherwise
        if ~isfield(p, 'beta')
            p.beta = 1;
        end
end

if p.random_seed > 0
    rand('seed', p.random_seed);
end

if ~isfield(p, 'init_w')
    if ~isfield(p, 'r')
        error('Number of components or initialization must be given')
    end
    r = p.r;
    w = rand(m, r);
else
    ri = size(p.init_w, 2);
    w(:, 1:ri) = p.init_w;
    if isfield(p, 'r') && ri < p.r
        w(:, (ri + 1) : p.r) = rand(m, p.r - ri);
        r = p.r;
    else
        r = ri;
    end
end
const_MLD_w = zeros(m, r); %Buffer initialization
const_MLD_h = zeros(r, n); %Buffer initialization
r_c = p.R_c;
c_num = round(r / r_c);
h_l1 = zeros(c_num, n);
h_l1_span = zeros(r, n);

if ~isfield(p, 'init_h')
    h = rand(r, n);
elseif ischar(p.init_h) && strcmp(p.init_h, 'ones')
    fprintf('sup_nmf: Initalizing H with ones.\n');
    h = ones(r, n);
else
    h = p.init_h;
end

if ~isfield(p, 'w_update_ind')
    p.w_update_ind = true(r, 1);
end

if ~isfield(p, 'h_update_ind')
    p.h_update_ind = true(r, 1);
end

% sparsity per matrix entry
if length(p.sparsity) == 1
    p.sparsity = ones(r, n) * p.sparsity;
elseif size(p.sparsity, 2) == 1
    p.sparsity = repmat(p.sparsity, 1, n);
end

% Normalize the columns of W and rescale H accordingly
wn = sqrt(sum(w.^2));
w  = bsxfun(@rdivide,w,wn);
h  = bsxfun(@times,  h,wn');

if ~isfield(p, 'display') 
    p.display = 0;
end

flr = 1e-9;
lambda = max(w * h, flr);
last_cost = Inf;
v= max(v, flr);
const_MLD_w = max(const_MLD_w, flr);

objective = struct;
objective.div = zeros(1,p.max_iter);
objective.cost = zeros(1,p.max_iter);

div_beta  = 1; %Fix to KL case
h_ind = p.h_update_ind;
w_ind = p.w_update_ind;
update_h = sum(h_ind);
update_w = sum(w_ind);

if p.display ~= 0
    fprintf(1,'Performing sparse NMF with beta-divergence, beta=%.1f\n',div_beta);
end

% tic
for it = 1:p.max_iter
    
    % W updates
    if update_w > 0
        for g = 1:c_num
            w_c = w(:, (g-1)*r_c+1 : g*r_c);
            h_c = h((g-1)*r_c+1 : g*r_c, :);
            mu_c = mu(:, (g-1)*r_c+1 : g*r_c);
            switch div_beta
                case 1
                    dpw = bsxfun(@plus,sum(h_c, 2)', ...
                        bsxfun(@times, ...
                        sum((v ./ lambda) * h_c' .* w_c), w_c));
                    dpw = max(dpw, flr);
                    dmw = v ./ lambda * h_c' ...
                        + bsxfun(@times, ...
                        sum(bsxfun(@times, sum(h_c,2)', w_c)), w_c);

                    %MLD constraints
                    const_MLD_w = p.sparsity_eta .* mu_c ./ w_c;
                    w_c = w_c .* (dmw + const_MLD_w) ./ (dpw + p.sparsity_eta);
            end
            w(:, (g-1)*r_c+1 : g*r_c) = w_c;
        end
        
        % Normalize the columns of W
        w = bsxfun(@rdivide,w,sqrt(sum(w.^2)));
        lambda = max(w * h, flr);
    end

    % H updates
    if update_h > 0
        switch div_beta
            case 1
                dph = bsxfun(@plus, sum(w(:, h_ind))', p.sparsity);
                dph = max(dph, flr);
                dmh = w(:, h_ind)' * (v ./ lambda);
                h(h_ind, :) = bsxfun(@rdivide, h(h_ind, :) .* dmh, dph);
                
                for g = 1:c_num
                   h_l1(g,:) = sum(h((g-1)*r_c+1 : g*r_c, :), 1);  
                end
                h_l1_span = kron(h_l1, ones(r_c,1));
                const_MLD_h(h_ind, :) = (1 + p.sparsity ./ (p.sparsity_epsilon + h_l1_span(h_ind, :)));
                h(h_ind, :) = h(h_ind, :) ./ const_MLD_h(h_ind, :);
        end
        lambda = max(w * h, flr);
    end    
    
    % Compute the objective function
    switch div_beta
        case 1
            div = sum(sum(v .* log(v ./ lambda) - v + lambda));
        case 2
            div = sum(sum((v - lambda) .^ 2));
        case 0
            div = sum(sum(v ./ lambda - log ( v ./ lambda) - 1)); 
        otherwise
            div = sum(sum(v.^div_beta + (div_beta - 1)*lambda.^div_beta ...
                - div_beta * v .* lambda.^(div_beta - 1))) / (div_beta * (div_beta - 1));
    end
    
    if p.cost_check
        cost = div + sum(sum(p.sparsity .* h));

        objective.div(it)  = div;
        objective.cost(it) = cost;

        if p.display ~= 0
            fprintf(repmat('\b',1,length(str)));
            str = sprintf('iteration %d div = %.3e cost = %.3e', it, div, cost);
            fprintf('%s', str);
        end

        % Convergence check
        if it > 1 && p.conv_eps > 0
            e = abs(cost - last_cost) / last_cost;
            if (e < p.conv_eps)
                if p.display ~= 0
                    disp('Convergence reached, aborting iteration')
                end
                objective.div = objective.div(1:it);
                objective.cost = objective.cost(1:it);
                break
            end
        end
        last_cost = cost;
    end
end
% toc
if p.display ~= 0
    disp('\nMax Iteration reached, aborting iteration\n');
end

end
