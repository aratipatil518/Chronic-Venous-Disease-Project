function simpleAdditionGUI
    % Create the main figure window
    fig = figure('Name', 'Simple Addition GUI', 'NumberTitle', 'off', 'Position', [100, 100, 300, 200]);

    % Create UI components
    uicontrol('Style', 'text', 'Position', [20, 150, 120, 25], 'String', 'Number 1:');
    num1Edit = uicontrol('Style', 'edit', 'Position', [150, 150, 100, 25]);

    uicontrol('Style', 'text', 'Position', [20, 120, 120, 25], 'String', 'Number 2:');
    num2Edit = uicontrol('Style', 'edit', 'Position', [150, 120, 100, 25]);

    resultText = uicontrol('Style', 'text', 'Position', [20, 90, 250, 25], 'String', 'Result:');

    addBtn = uicontrol('Style', 'pushbutton', 'Position', [20, 50, 100, 30], 'String', 'Add Numbers', 'Callback', @addNumbers);

    % Callback function for the button click
    function addNumbers(~, ~)
        % Get the values from the edit fields
        num1 = str2double(get(num1Edit, 'String'));
        num2 = str2double(get(num2Edit, 'String'));

        % Check if the input is valid
        if isnan(num1) || isnan(num2)
            errordlg('Please enter valid numbers.', 'Error', 'modal');
            return;
        end

        % Perform addition
        result = num1 + num2;

        % Display the result
        set(resultText, 'String', ['Result: ' num2str(result)]);
    end
end
