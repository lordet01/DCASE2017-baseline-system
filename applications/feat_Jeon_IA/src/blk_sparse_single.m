function [Q]=blk_sparse_single(X, p)

[K,T] = size(X);

% Method1: Q. Hoyer
feat = X;
feat = max(feat, eps);
Q = zeros(1,T);
n =K;
for t = 1:T
    b = feat(:,t);
    b = b ./ max(b);
    feat_l1 = sum(b, 1);
    feat_l2 = sqrt(sum(b.^2, 1));
    Q(1,t) = (sqrt(n) - feat_l1 ./ feat_l2) / (sqrt(n)-1); %Method1: Q. Hoyer, 2004
end

end
