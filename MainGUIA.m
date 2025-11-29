%%
% 
%   for x = 1:10
%       disp(x)
%   end
% 
function MainGUIA
    % Create figure
    %f = figure('Name', 'Project Title', 'NumberTitle', 'off');
    f = figure('Name', 'Chronic Venous Disease Classification', 'NumberTitle', 'off', 'MenuBar', 'none');

     % Shared variables
    imds = [];
    imdsTrain = [];
    %%
    % 
    % $$e^{\pi i} + 1 = 0$$
    % 
    imdsTest = [];
    layers = [];
    options = [];
    net = [];
    imagePath="";

    % Set figure size and position
    f.Position = [100, 100, 800, 500];
    f.Color = [0.9, 0.9, 0.9];  % Set background color

    % Title Bar
    titleBar = uicontrol('Style', 'text', 'String', 'Chronic Venous Disease Classification', 'FontWeight', 'bold', 'FontSize', 24,'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [0.2, 0.2, 0.2]);
    titleBar.Position = [10, 450, 780, 40];

    % Preprocess Button
    preprocessButton = uicontrol('Style', 'pushbutton', 'String', 'Preprocess', 'Position', [50, 380, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @preprocessCallback);

    % Train Test Split Button
    trainTestSplitButton = uicontrol('Style', 'pushbutton', 'String', 'Train Test Split', 'Position', [50, 320, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @trainTestSplitCallback);

    % Train Data Button
    trainDataButton = uicontrol('Style', 'pushbutton', 'String', 'Train Data', 'Position', [50, 260, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @trainDataCallback);

    % Analysis and Test Button
    analysisTestButton = uicontrol('Style', 'pushbutton', 'String', 'Analysis and Test', 'Position', [50, 200, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @analysisTestCallback);

    % Save Model Button
    saveModelButton = uicontrol('Style', 'pushbutton', 'String', 'Save Model', 'Position', [50, 140, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @saveModelCallback);

    % Select Image Button
    LoadModelButton = uicontrol('Style', 'pushbutton', 'String', 'Load Model', 'Position', [400, 380, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @LoadModelCallback);

    % Select Image Button
    selectImageButton = uicontrol('Style', 'pushbutton', 'String', 'Select Image', 'Position', [400, 320, 200, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @selectImageCallback);

    % Input Text
    inputText = uicontrol('Style', 'edit', 'String', '', 'Position', [400, 260, 320, 40], 'FontSize', 12, 'BackgroundColor', [1, 1, 1]);

    % Show Image Button
    showImageButton = uicontrol('Style', 'pushbutton', 'String', 'Show Image', 'Position', [400, 200, 150, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @showImageCallback);

    % Predict Button
    predictButton = uicontrol('Style', 'pushbutton', 'String', 'Predict', 'Position', [560, 200, 150, 40], 'FontSize', 14, 'BackgroundColor', [0.2, 0.6, 0.8], 'ForegroundColor', [1, 1, 1], 'Callback', @predictCallback);

    % Result Label
    resultLabel = uicontrol('Style', 'text', 'String', '', 'Position', [400, 140, 320, 40], 'FontSize', 14, 'BackgroundColor', [1, 1, 1], 'HorizontalAlignment', 'center');

    % Callback Functions
    function preprocessCallback(~, ~)
        disp('Preprocessing...');
          % Specify the path to your dataset folder
        datasetPath = 'imagedata';
        
        % Create an imageDatastore for your dataset
        imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        % Display a message indicating completion
        disp('Image Preprocessing Completed');
    end

    function trainTestSplitCallback(~, ~)
        disp('Train Test Splitting...');
         % Check if imds is empty
        if isempty(imds)
            disp('Please preprocess images first.');
            return;
        end
        
        % Split the dataset into training and testing sets
        [imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
        
        % Display a message indicating completion
        disp('Training and Testing Split Completed');
    end

    function trainDataCallback(~, ~)
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
            'MaxEpochs', 35, ...
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'InitialLearnRate', 0.001, ...
            'Plots', 'training-progress');
        
        % Train the CNN model net
        %net = trainNetwork(imdsTrain, layers, options);
        [net, trainingInfo] = trainNetwork(imdsTrain, layers, options);

        % Save training information
        save('trainingInfo.mat', 'trainingInfo');
        
        save('trained_model.mat', 'net');

        % Display a message indicating completion
        disp('Training Completed');
    end

    function analysisTestCallback(~, ~)
        disp('Analyzing and Testing...');
         % Check if the training has been completed
        if isempty(net)
            disp('Please train the model first.');
            return;
        end
    
        % Evaluate the trained network using testing data
        YPred = classify(net, imdsTest);
        % Display list of all imdsTest with their predictions
        disp('List of all imdsTest with their predictions:');
        disp([imdsTest.Files, YPred]);


        % Calculate accuracy
        accuracy = mean(YPred == imdsTest.Labels);
        
        % Display accuracy
        disp(['Accuracy using testing data: ', num2str(accuracy)]);

    end

    function saveModelCallback(~, ~)
        disp('Saving Model...');
        % Save the trained model
        save('trained_model.mat', 'net');
    end

    function LoadModelCallback(~, ~)
        disp('Load Model...');
        loaded_model = load('trained_model.mat');
        net = loaded_model.net;
    end

    function selectImageCallback(~, ~)
        [fileName, filePath] = uigetfile({'*.jpg;*.bmp;*.png', 'Image Files (*.jpg,*.bmp, *.png)'}, 'Select Image');
        if isequal(fileName, 0) || isequal(filePath, 0)
            disp('No file selected.');
        else
            imagePath = fullfile(filePath, fileName);
            inputText.String = imagePath;
            disp(['Image selected: ', imagePath]);
        end
    end

    function showImageCallback(~, ~)
        imagePath = inputText.String;
        if isempty(imagePath)
            disp('No image selected.');
        else
            img = imread(imagePath);
            showImageInNewFigure(img);
        end
    end

    function predictCallback(~, ~)
        disp('Predicting...');

        img = imread(imagePath);
        YPred = classify(net, img);
        disp(['Predicted label: ', char(YPred)]);

        % Perform prediction based on the selected image
        %resultLabel.String = 'Prediction result-'+char(YPred);

        % Perform prediction based on the selected image
        if(char(YPred)=='1')
            resultLabel.String = char('Normal Skin ');
        elseif(char(YPred)=='2')
            resultLabel.String = char('Reticular vein or Telangiectasia');
        elseif(char(YPred)=='3')
            resultLabel.String = char('Varicose Veins');
        elseif(char(YPred)=='4')
            resultLabel.String = char('Pigmentation or Edema');
        elseif(char(YPred)=='5')
            resultLabel.String = char('Venous Ulcers');
        else
            resultLabel.String = char('No detected');
        end
    end

    function showImageInNewFigure(img)
        fig = figure('Name', 'Image Preview');
        ax = axes('Parent', fig);
        imshow(img, 'Parent', ax);
        axis(ax, 'image');
        title(ax, 'Selected Image');
    end
end

