function [ accuracy ] = ComputeAcc(P, labels)
%Compute accuracy given probability matrix and the true labels
%INPUT:
%               P:          P(y|x) for each x and each y, a n_row * n_class matrix     
%               labels:     true class for each x, a n_row * 1 vector
%       
%OUTPUT:       
%               accuracy
         
[ n_row, n_class ] = size(P);
if n_class ~= 2
    [ ~, maxind ] = max(P, [], 2);
    accuracy = sum(maxind == labels)/n_row;
else
    [ ~, maxind ] = max(P, [], 2 );
    accuracy = (sum((maxind == 1) & (labels == 1)) + sum((maxind == 2) & (labels == -1)))/n_row;
end

end

