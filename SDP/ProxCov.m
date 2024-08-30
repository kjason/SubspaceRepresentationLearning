function T = ProxCov(Y,S,epsilon)
n = size(S,1);
m = size(S,2);
l = size(Y,2);
eI = epsilon*eye(l);
cvx_begin sdp quiet
    variable T(m,m) hermitian toeplitz
    variable W(l,l) hermitian complex
    minimize( norm(Y*W*(Y')-S*T*(S'),'fro') )
    T >= 0;
    W >= eI;
cvx_end
end