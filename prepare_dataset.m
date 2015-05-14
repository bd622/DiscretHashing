function prepare_dataset(dataset)
% dataset is stored in a row-wise matrix
%%
load(['./datasets/',dataset]);

% Normalize all feature vectors to unit length
traindata = normalize(double(traindata));
testdata  = normalize(double(testdata));

cateTrainTest = bsxfun(@eq, traingnd, testgnd'); % traingnd and testgnd are the labels.

save(['testbed/',dataset],'traindata','testdata','traingnd','testgnd','cateTrainTest', '-v7.3');

clear;


