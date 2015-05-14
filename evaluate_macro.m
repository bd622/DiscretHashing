function [p, r] = evaluate_macro(Rel, Ret)
% evaluate macro_averaged performance
% Input:
%    Rel = relevant  train documents for each test document
%    Ret = retrieved train documents for each test document
% Output:
%    p   = macro-averaged precision
%    r   = macro-averaged recall

numTest = size(Rel,2);
precisions = zeros(1,numTest);
recalls    = zeros(1,numTest);

retrieved_relevant_pairs = (Rel & Ret);

for j = 1:numTest
    retrieved_relevant_num = nnz(retrieved_relevant_pairs(:,j));
    retrieved_num = nnz(Ret(:,j));
    relevant_num  = nnz(Rel(:,j));
    if retrieved_num
        precisions(j) = retrieved_relevant_num / retrieved_num;
    else
        precisions(j) = 0;
    end
    if relevant_num
        recalls(j) = retrieved_relevant_num / relevant_num;
    else
        recalls(j) = 0;
    end
end

p = mean(precisions);
r = mean(recalls);

end
