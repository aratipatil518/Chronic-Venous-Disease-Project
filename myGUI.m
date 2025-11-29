function myGUI
    % Create the main figure window
    fig = figure('Name', 'My GUI', 'NumberTitle', 'off', 'Position', [100, 100, 400, 300]);

    % Shared variables
    imds = [];
    imdsTrain = [];
    layers = [];
    options = [];
    net = [];

    % Button 1: Image Preprocessing
    btn1 = uicontrol('Style', 'pushbutton', 'Position', [20, 230, 150, 30], 'String', 'Image Preprocessing', 'Callback', @imagePreprocessing);

    % Button 2: Training and Testing Split
    btn2 = uicontrol('Style', 'pushbutton', 'Position', [20, 190, 150, 30], 'String', 'Training and Testing Split', 'Callback', @trainTestSplit);

    % Button 3: Train Data
    btn3 = uicontrol('Style', 'pushbutton', 'Position', [20, 150, 150, 30], 'String', 'Train Data', 'Callback', @trainData);

    % Button 4: Loss and Accuracy Graph
    btn4 = uicontrol('Style', 'pushbutton', 'Position', [20, 110, 150, 30], 'String', 'Loss and Accuracy Graph', 'Callback', @plotGraphs);

    % Button 1 Callback: Image Preprocessing
    function imagePreprocessing(~, ~)
        % Specify the path to your dataset folder
        datasetPath = 'imagedata';
        
        % Create an imageDatastore for your dataset
        imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        % Display a message indicating completion
        disp('Image Preprocessing Completed');
    end

    % Button 2 Callback: Training and Testing Split
    function trainTestSplit(~, ~)
        % Check if imds is empty
        if isempty(imds)
            disp('Please preprocess images first.');
            return;
        end
        
        % Split the dataset into training and testing sets
        [imdsTrain, ~] = splitEachLabel(imds, 0.8, 'randomized');
        
        % Display a message indicating completion
        disp('Training and Testing Split Completed');
    end

    % Button 3 Callback: Train Data
    function trainData(~, ~)
        % Check if imdsTrain is empty
        if isempty(imdsTrain)
            disp('Please perform training and testing split first.');
            return;
        end

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
            'MaxEpochs', 30, ...
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'InitialLearnRate', 0.001, ...
            'Plots', 'training-progress');
        
        % Train the CNN model net
        %net = trainNetwork(imdsTrain, layers, options);
        [net, trainingInfo] = trainNetwork(imdsTrain, layers, options);

        % Save training information
        save('trainingInfo.mat', 'trainingInfo');
        
        
        % Display a message indicating completion
        disp('Training Completed');
    end

    % Function to save training information during the training process
    function stop = saveTrainingInfo(info)
        % Save training information to a file
        save(fullfile(pwd, 'trainingInfo.mat'), 'info', '-append');
    
        % Check if the training should stop
        stop = false;  % Continue training
    end

   % Button 4 Callback: Loss and Accuracy Graph
    function plotGraphs(~, ~)
        % Check if the training has been completed
        if isempty(net)
            disp('Please train the model first.');
            return;
        end
    
        % Extract training information from the training history
        %trainingInfo = net.Layers(end).ExecutionInfo;
        %trainingInfo = net.LearnInfo;

        % Load training information
        load('trainingInfo.mat', 'info');

    
        % Plot the loss and accuracy graphs
        figure;
    
        % Plot training loss
        subplot(2, 1, 1);
        plot([trainingInfo.TrainingLoss]);
        title('Training Loss');
        xlabel('Epoch');
        ylabel('Loss');
    
        % Plot training accuracy
        subplot(2, 1, 2);
        plot([trainingInfo.TrainingAccuracy]);
        title('Training Accuracy');
        xlabel('Epoch');
        ylabel('Accuracy');
    end
end
