function R=StructCovMLE_noisevar(S,So,epsilon,mmITER,lambda1)
% Modified from https://github.com/rohanpote/GridlessDoA_StructCovMLE/blob/main/lib/StructCovMLE_MUSIC.m
% Originally written by Rohan R. Pote, 2022
% Pote, Rohan R., and Bhaskar D. Rao.
% "Maximum likelihood-based gridless DoA estimation using structured
% covariance matrix recovery and SBL with grid refinement."
% IEEE Transactions on Signal Processing 71 (2023): 802-815.
M=size(So,1);
Mapt=size(So,2);
x_old = [1 zeros(1,Mapt-1)];
for mmloop = 1:mmITER
    B0 = inv(So*toeplitz(x_old)*So'+lambda1*eye(M));
    cvx_begin sdp quiet
    %     cvx_solver sedumi
    variable x(1,Mapt) complex
    variable U(M,M) hermitian
    minimize(real(trace(B0*(So*toeplitz(x)*So')))+real(trace(U*S)))
    subject to
    (toeplitz(x)+toeplitz(x)')/2>=0
    [U eye(M); eye(M) So*((toeplitz(x)+toeplitz(x)')/2)*So'+lambda1*eye(M)]>=0
    cvx_end
    relative_change = norm(x-x_old,'fro') / norm(x_old,'fro');
    if relative_change < epsilon
        break
    end
    x_old = x;
end
R = toeplitz(x);
end