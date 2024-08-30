function R0 = Wasserstein(R_hat,S)
n = size(S,1);
m = size(S,2);
R_hat = 0.5*(R_hat+R_hat');
cvx_begin sdp quiet
    variable R0(m,m) hermitian toeplitz
    variable V(n,n) complex
    minimize( trace( R_hat + S*R0*(S') - V - V') )
    [S*R0*(S'), V;V', R_hat] >= 0;
    R0 >= 0;
cvx_end
end