function M = CS_GeometricMean(A, B, t)
%A = A + lambda*eye(size(A,1));
%B = B + lambda*eye(size(B,1));
cond_A = norm(A)/norm(inv(A));
cond_B = norm(B)/norm(inv(B));
if (cond_A>cond_B)
    R_B=chol(B);
    V=(inv(R_B))'*A*inv(R_B);
    V=(V+V')/2;
    [U, D]= schur(V);
    M= (R_B)'*U*D^(1-t)*U'*R_B;
    M=(M+M')/2;
else
    R_A= chol(A);
    V=(inv(R_A))'*B*inv(R_A);
    V=(V+V')/2;
    [U, D]= schur(V);
    M= (R_A)'*U*D^(t)*U'*R_A;
    M=(M+M')/2;
end
end

