
% Specify the path to your dataset folder
datasetPath = 'imagedata';
% Create an imageDatastore for your dataset
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the dataset into training and testing sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Define CNN architecture
layers = [
    imageInputLayer([700 250 3])  % Adjust the input size based on your images
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(5)  % Adjust the output size based on the number of classes
    softmaxLayer
    classificationLayer
];
% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001, ...
    'Plots', 'training-progress');
% Train the CNN model net
net = trainNetwork(imdsTrain, layers, options);


% Evaluate the trained network using testing data
YPred = classify(net, imdsTest);

% Calculate accuracy
accuracy = mean(YPred == imdsTest.Labels);

% Display accuracy
disp(['Accuracy using testing data: ', num2str(accuracy)]);

 % Display list of all imdsTest with their predictions
disp('List of all imdsTest with their predictions:');
disp([imdsTest.Files, YPred]);

% Set one image as input and predict
inputImage = imread('/MATLAB Drive/imagedata/1/0001.bmp'); % Replace 'path_to_input_image.jpg' with the path to your input image
YPredInput = classify(net, inputImage);
disp('Prediction for the input image:');
disp(YPredInput);



% Save the trained model
save('trained_model.mat', 'net');

% Load the saved model
loaded_net = load('trained_model.mat');
net = loaded_net.net;

% Set one image as input and predict
imagePath = '/MATLAB Drive/imagedata/1/0001.bmp';
img = imread(imagePath);
YPred = classify(net, img);
disp(['Predicted label: ', char(YPred)]);