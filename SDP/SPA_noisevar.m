function T = SPA_noisevar(R_hat,S,lambda)
n = size(S,1);
m = size(S,2);
R_hat_inv = inv(R_hat);
R_hat_inv = 0.5*(R_hat_inv+R_hat_inv');
R_sqrt = sqrtm(R_hat);
Z = zeros(n,m);
cvx_begin sdp quiet
    variable T(m,m) hermitian toeplitz
    variable X(n,n) hermitian complex
    minimize( real(trace(X) + trace(R_hat_inv*S*T*(S'))) )
    [X, R_sqrt, Z; R_sqrt', S*T*(S')+lambda*eye(n), Z; Z', Z', T] >= 0;
cvx_end
end