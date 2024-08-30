function V = StructCovMLE(R_hat,S,epsilon,max_iter)
% """
% Created on Sun Jun 9 2024
%
% @author: Kuan-Lin Chen
%
% Implementation of the StructCovMLE approach in the following paper:
%
% Pote, Rohan R., and Bhaskar D. Rao.
% "Maximum likelihood-based gridless DoA estimation using structured
% covariance matrix recovery and SBL with grid refinement."
% IEEE Transactions on Signal Processing 71 (2023): 802-815.
%
% @param R_hat: the sample spatial covariance matrix received at the sparse linear array
% @param S: the row-selection matrix containing only ones and zeros
% @param epsilon: the threshold of relative change (one of the stopping criterion)
% @param max_iter: the maximum number of iterations (one if the stopping criterion)
% @return V: the spatial covariance matrix estimate of the corresponding ULA
% """
n = size(S,1);
m = size(S,2);
I = eye(n);
V = eye(m);
for i = 1:max_iter
    V_prev = V;
    Vs_inv = inv(S*V*(S'));
    Vs_inv = 0.5*(Vs_inv+Vs_inv');
    Z = zeros(n,m);
    cvx_begin sdp quiet
    variable T(m,m) hermitian toeplitz
    variable X(n,n) hermitian complex
    minimize( real(trace(Vs_inv*S*T*(S')) + trace(X*R_hat)) )
    [X, I, Z; I, S*T*(S'), Z; Z', Z', T] >= 0;
    cvx_end
    V = T;
    relative_change = norm(V-V_prev,'fro') / norm(V_prev,'fro');
    if relative_change < epsilon
        break
    end
end
end