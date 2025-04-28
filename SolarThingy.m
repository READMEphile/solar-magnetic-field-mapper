% Solar Magnetic Field Mapper with Machine Learning
% This program downloads and visualizes solar magnetogram data from SDO
% and provides tools for tracking sunspots and magnetic field changes over time
% Includes ML-based solar flare prediction

clear all; close all; clc;

%% Configuration Parameters
fprintf('Solar Magnetic Field Mapper initializing...\n');

% Data sources - NASA SDO (Solar Dynamics Observatory) HMI instrument
sdo_base_url = 'https://sdo.gsfc.nasa.gov/assets/img/browse';
start_date = '2023-01-01'; % Default start date
end_date = '2023-01-15';   % Default end date (two week period for ML training)

% Data directory
data_dir = 'solar_data';
ml_model_file = 'flare_prediction_model.mat';

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf('Created data directory: %s\n', data_dir);
end

%% User Interface for Data Selection
fprintf('\nSolar Magnetic Field Mapper\n');
fprintf('=========================\n\n');
fprintf('1. Download recent magnetogram data\n');
fprintf('2. Load saved magnetogram data\n');
fprintf('3. Visualize magnetic field over time\n');
fprintf('4. Analyze sunspot activity\n');
fprintf('5. Train ML flare prediction model\n');
fprintf('6. Predict solar flares using ML\n');
fprintf('7. Exit\n\n');

choice = input('Enter your choice (1-7): ');

%% Main Program Logic
switch choice
    case 1
        downloadMagnetogramData(data_dir, sdo_base_url, start_date, end_date);
    case 2
        data = loadMagnetogramData(data_dir);
    case 3
        data = loadMagnetogramData(data_dir);
        visualizeMagneticField(data);
    case 4
        data = loadMagnetogramData(data_dir);
        analyzeSunspotActivity(data);
    case 5
        data = loadMagnetogramData(data_dir);
        trainFlareModel(data, ml_model_file);
    case 6
        data = loadMagnetogramData(data_dir);
        if ~exist(ml_model_file, 'file')
            fprintf('ML model not found. Training new model first...\n');
            trainFlareModel(data, ml_model_file);
        end
        predictSolarFlaresML(data, ml_model_file);
    case 7
        fprintf('Exiting program.\n');
        return;
    otherwise
        fprintf('Invalid choice. Please restart the program.\n');
end

%% Function to download magnetogram data from SDO
function downloadMagnetogramData(data_dir, sdo_base_url, start_date, end_date)
    fprintf('Downloading magnetogram data from %s to %s...\n', start_date, end_date);
    
    % Convert dates to datetime objects for iteration
    start_dt = datetime(start_date);
    end_dt = datetime(end_date);
    
    % Iterate through each day in the range
    date_range = start_dt:end_dt;
    for i = 1:length(date_range)
        current_date = date_range(i);
        year_str = num2str(year(current_date));
        month_str = sprintf('%02d', month(current_date));
        day_str = sprintf('%02d', day(current_date));
        
        % Create URL for HMI magnetogram (Helioseismic and Magnetic Imager)
        url = sprintf('%s/%s/%s/%s/HMI_magnetogram_2048.jpg', ...
                     sdo_base_url, year_str, month_str, day_str);
        
        % Create local filename
        filename = fullfile(data_dir, sprintf('magnetogram_%s_%s_%s.jpg', ...
                           year_str, month_str, day_str));
        
        try
            fprintf('Downloading data for %s-%s-%s... ', year_str, month_str, day_str);
            
            % In a real application, we would use websave:
            % websave(filename, url);
            
            % For this example, we'll simulate downloading by creating a simulated magnetogram
            createSimulatedMagnetogram(filename, current_date);
            
            fprintf('Done!\n');
        catch e
            fprintf('Failed: %s\n', e.message);
        end
    end
    
    fprintf('Download complete. Data saved to %s\n', data_dir);
end

%% Function to load magnetogram data from files
function data = loadMagnetogramData(data_dir)
    fprintf('Loading magnetogram data...\n');
    
    % Get list of magnetogram files
    files = dir(fullfile(data_dir, 'magnetogram_*.jpg'));
    
    if isempty(files)
        fprintf('No magnetogram data found. Please download data first.\n');
        data = [];
        return;
    end
    
    % Load and process each file
    for i = 1:length(files)
        filename = fullfile(data_dir, files(i).name);
        
        % Extract date from filename
        date_parts = sscanf(files(i).name, 'magnetogram_%d_%d_%d.jpg');
        if length(date_parts) >= 3
            data(i).date = datetime(date_parts(1), date_parts(2), date_parts(3));
        else
            data(i).date = datetime('now');
        end
        
        % Load image
        fprintf('Loading %s...\n', files(i).name);
        data(i).magnetogram = imread(filename);
        
        % For real magnetograms, this would convert the image to actual magnetic field values
        % For this simulation, we'll just use the grayscale values as mock field strengths
        if size(data(i).magnetogram, 3) == 3  % If RGB, convert to grayscale
            data(i).field_strength = double(rgb2gray(data(i).magnetogram));
        else
            data(i).field_strength = double(data(i).magnetogram);
        end
        
        % Scale values to represent magnetic field in gauss (simulated)
        data(i).field_strength = (data(i).field_strength - 128) * 40;  % Scale to approx +/- 5000 gauss
        
        % Extract features for ML
        data(i).features = extractFeatures(data(i).field_strength);
        
        % Simulate flare observations for training data
        % In real application, this would come from GOES X-ray flux data
        % For simulation: probability based on complexity plus some randomness
        complexity = data(i).features.gradient_complexity / 50 + ...
                    data(i).features.polarity_inversion_length / 10000;
        
        % Add simulated flare observations for the next 24 hours
        % Class: 0=None, 1=C-class, 2=M-class, 3=X-class
        r = rand();
        if complexity < 0.3
            if r < 0.95
                data(i).flare_class = 0;  % No flare (95% if low complexity)
            else
                data(i).flare_class = 1;  % C-class (5% chance)
            end
        elseif complexity < 0.6
            if r < 0.7
                data(i).flare_class = 0;  % No flare (70%)
            elseif r < 0.9
                data(i).flare_class = 1;  % C-class (20%)
            else
                data(i).flare_class = 2;  % M-class (10%)
            end
        elseif complexity < 0.8
            if r < 0.4
                data(i).flare_class = 0;  % No flare (40%)
            elseif r < 0.7
                data(i).flare_class = 1;  % C-class (30%)
            elseif r < 0.9
                data(i).flare_class = 2;  % M-class (20%)
            else
                data(i).flare_class = 3;  % X-class (10%)
            end
        else
            if r < 0.2
                data(i).flare_class = 0;  % No flare (20%)
            elseif r < 0.4
                data(i).flare_class = 1;  % C-class (20%)
            elseif r < 0.7
                data(i).flare_class = 2;  % M-class (30%)
            else
                data(i).flare_class = 3;  % X-class (30%)
            end
        end
    end
    
    fprintf('Loaded %d magnetogram files.\n', length(files));
end

%% Function to extract features for machine learning
function features = extractFeatures(field_strength)
    % Calculate gradient magnitude and complexity
    [gx, gy] = gradient(field_strength);
    gradient_magnitude = sqrt(gx.^2 + gy.^2);
    features.gradient_complexity = mean(gradient_magnitude(:));
    features.gradient_max = max(gradient_magnitude(:));
    
    % Calculate total unsigned magnetic flux
    features.total_unsigned_flux = sum(abs(field_strength(:)));
    
    % Calculate net magnetic flux (signed)
    features.net_flux = sum(field_strength(:));
    
    % Identify strong field regions
    strong_field_pos = field_strength > 1000;
    strong_field_neg = field_strength < -1000;
    features.strong_field_area = sum(strong_field_pos(:)) + sum(strong_field_neg(:));
    
    % Identify polarity inversion lines
    field_smooth = imgaussfilt(field_strength, 5);  % Smoothing to reduce noise
    pos_field = field_smooth > 200;
    neg_field = field_smooth < -200;
    
    % Dilate opposite polarities
    se = strel('disk', 5);
    pos_dilated = imdilate(pos_field, se);
    neg_dilated = imdilate(neg_field, se);
    
    % Find where they overlap (polarity inversion lines)
    inversion_lines = pos_dilated & neg_dilated;
    features.polarity_inversion_length = sum(inversion_lines(:));
    
    % Calculate proximity of strong opposite polarity regions
    dist_transform_pos = bwdist(strong_field_pos);
    dist_transform_neg = bwdist(strong_field_neg);
    
    % Average distance from positive to negative regions and vice versa
    avg_dist_pos_to_neg = mean(dist_transform_neg(strong_field_pos));
    avg_dist_neg_to_pos = mean(dist_transform_pos(strong_field_neg));
    
    if isnan(avg_dist_pos_to_neg)
        avg_dist_pos_to_neg = size(field_strength, 1); % If no positive regions
    end
    if isnan(avg_dist_neg_to_pos)
        avg_dist_neg_to_pos = size(field_strength, 1); % If no negative regions
    end
    
    features.polarity_proximity = 1 / (1 + min(avg_dist_pos_to_neg, avg_dist_neg_to_pos));
    
    % Calculate field complexity using entropy
    field_norm = (field_strength - min(field_strength(:))) / (max(field_strength(:)) - min(field_strength(:)));
    field_hist = histcounts(field_norm(:), 20);
    field_hist_norm = field_hist / sum(field_hist);
    entropy_value = -sum(field_hist_norm .* log2(field_hist_norm + eps));
    features.field_entropy = entropy_value;
    
    % Calculate fractal dimension (box-counting) of strong field regions
    binary_field = abs(field_strength) > 500;
    features.fractal_dimension = calculateFractalDimension(binary_field);
    
    % Calculate asymmetry between positive and negative regions
    pos_flux = sum(field_strength(field_strength > 0));
    neg_flux = abs(sum(field_strength(field_strength < 0)));
    features.flux_asymmetry = abs(pos_flux - neg_flux) / (pos_flux + neg_flux + eps);
end

%% Function to calculate fractal dimension using box-counting method
function fd = calculateFractalDimension(binary_image)
    % Simple implementation of box-counting algorithm
    s = size(binary_image);
    
    % Make sure the image is square and size is power of 2
    max_size = 2^floor(log2(min(s)));
    if max_size < 8
        fd = 1; % Default value if image is too small
        return;
    end
    
    % Crop to power of 2 size
    binary_image = binary_image(1:max_size, 1:max_size);
    
    % Box counting for different box sizes
    box_sizes = 2.^(0:floor(log2(max_size/2)));
    counts = zeros(size(box_sizes));
    
    for i = 1:length(box_sizes)
        box_size = box_sizes(i);
        grid = false(max_size/box_size, max_size/box_size);
        
        % Check each box for presence of object
        for row = 1:max_size/box_size
            for col = 1:max_size/box_size
                r_start = (row-1)*box_size + 1;
                r_end = row*box_size;
                c_start = (col-1)*box_size + 1;
                c_end = col*box_size;
                box_region = binary_image(r_start:r_end, c_start:c_end);
                grid(row, col) = any(box_region(:));
            end
        end
        
        counts(i) = sum(grid(:));
    end
    
    % Calculate fractal dimension from log-log plot
    valid_indices = counts > 0;
    if sum(valid_indices) < 2
        fd = 1; % Default value if not enough data points
        return;
    end
    
    log_sizes = log(1./box_sizes(valid_indices));
    log_counts = log(counts(valid_indices));
    
    % Linear regression to find slope
    p = polyfit(log_sizes, log_counts, 1);
    fd = p(1); % Slope is the fractal dimension
end

%% Function to visualize magnetic field data
function visualizeMagneticField(data)
    if isempty(data)
        fprintf('No data available for visualization.\n');
        return;
    end
    
    fprintf('Visualizing magnetic field data...\n');
    
    % Create figure for visualization
    figure('Name', 'Solar Magnetic Field Visualization', 'Position', [100, 100, 1200, 800]);
    
    % Number of data points
    num_frames = length(data);
    
    % Display each magnetogram with field lines
    for i = 1:num_frames
        % Plot the magnetogram
        subplot(2, 3, [1 2 4 5]);
        imagesc(data(i).field_strength);
        colormap(customMagnetogramColormap());
        colorbar('TickLabels',{'-5000G','-2500G','0','2500G','5000G'});
        title(sprintf('Solar Magnetogram - %s', datestr(data(i).date)), 'FontSize', 14);
        axis image off;
        hold on;
        
        % Calculate and overlay magnetic field lines
        [fx, fy] = gradient(data(i).field_strength);
        
        % Subsample for clearer visualization
        step = 20;
        [X, Y] = meshgrid(1:step:size(data(i).field_strength, 2), 1:step:size(data(i).field_strength, 1));
        fx_sub = fx(1:step:end, 1:step:end);
        fy_sub = fy(1:step:end, 1:step:end);
        
        % Normalize and plot as streamlines
        magnitude = sqrt(fx_sub.^2 + fy_sub.^2);
        fx_norm = fx_sub ./ (magnitude + eps);  % Add eps to avoid division by zero
        fy_norm = fy_sub ./ (magnitude + eps);
        
        streamslice(X, Y, fx_norm, fy_norm, 2, 'Arrow');
        hold off;
        
        % Plot active regions histogram
        subplot(2, 3, 3);
        histogram(data(i).field_strength(abs(data(i).field_strength) > 500), 50);
        title('Active Regions Field Strength', 'FontSize', 12);
        xlabel('Magnetic Field (Gauss)');
        ylabel('Pixel Count');
        
        % Plot time series of total magnetic flux if multiple frames
        subplot(2, 3, 6);
        total_flux = zeros(num_frames, 1);
        for j = 1:num_frames
            total_flux(j) = sum(sum(abs(data(j).field_strength)));
        end
        
        dates = [data.date];
        plot(dates, total_flux, 'b-o', 'LineWidth', 2);
        hold on;
        plot(data(i).date, total_flux(i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold off;
        title('Total Magnetic Flux Over Time', 'FontSize', 12);
        xlabel('Date');
        ylabel('Total Flux (arb. units)');
        grid on;
        
        % Pause to display (would be removed for saving as video/animation)
        pause(0.5);
        
        % For single image, break after first display
        if num_frames == 1
            break;
        end
    end
    
    fprintf('Visualization complete.\n');
end

%% Function to analyze sunspot activity
function analyzeSunspotActivity(data)
    if isempty(data)
        fprintf('No data available for analysis.\n');
        return;
    end
    
    fprintf('Analyzing sunspot activity...\n');
    
    % Create figure for analysis
    figure('Name', 'Sunspot Activity Analysis', 'Position', [150, 150, 1200, 800]);
    
    % Process each magnetogram to identify sunspots
    num_frames = length(data);
    sunspot_count = zeros(num_frames, 1);
    sunspot_area = zeros(num_frames, 1);
    
    for i = 1:num_frames
        % Threshold the magnetic field to identify strong field regions
        threshold = 1000;  % Field strength threshold for sunspot identification in gauss
        strong_field = abs(data(i).field_strength) > threshold;
        
        % Use morphological operations to clean up the binary image
        se = strel('disk', 3);
        strong_field_cleaned = imclose(strong_field, se);
        strong_field_cleaned = imopen(strong_field_cleaned, se);
        
        % Label connected components (sunspot regions)
        [labeled, num_spots] = bwlabel(strong_field_cleaned);
        stats = regionprops(labeled, 'Area', 'Centroid');
        
        % Store results
        sunspot_count(i) = num_spots;
        if num_spots > 0
            sunspot_area(i) = sum([stats.Area]);
        end
        
        % Display results for this frame
        subplot(2, 2, 1);
        imagesc(data(i).field_strength);
        colormap(customMagnetogramColormap());
        colorbar;
        title(sprintf('Magnetogram - %s', datestr(data(i).date)), 'FontSize', 14);
        axis image off;
        
        subplot(2, 2, 2);
        imagesc(labeled);
        colormap(jet);
        title(sprintf('Identified Sunspots: %d', num_spots), 'FontSize', 14);
        axis image off;
        
        % Hold for current frame's spots
        hold on;
        for j = 1:num_spots
            text(stats(j).Centroid(1), stats(j).Centroid(2), ...
                 sprintf('%d', j), 'Color', 'w', 'FontWeight', 'bold');
        end
        hold off;
        
        % Time series plots
        dates = [data.date];
        
        subplot(2, 2, 3);
        plot(dates, sunspot_count, 'b-o', 'LineWidth', 2);
        hold on;
        plot(data(i).date, sunspot_count(i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold off;
        title('Sunspot Count Over Time', 'FontSize', 12);
        xlabel('Date');
        ylabel('Number of Sunspots');
        grid on;
        
        subplot(2, 2, 4);
        plot(dates, sunspot_area, 'g-o', 'LineWidth', 2);
        hold on;
        plot(data(i).date, sunspot_area(i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold off;
        title('Total Sunspot Area Over Time', 'FontSize', 12);
        xlabel('Date');
        ylabel('Area (pixels)');
        grid on;
        
        % Pause to display (would be removed for saving as video/animation)
        pause(0.5);
        
        % For single image, break after first display
        if num_frames == 1
            break;
        end
    end
    
    fprintf('Sunspot analysis complete.\n');
    fprintf('Average number of sunspots: %.2f\n', mean(sunspot_count));
    fprintf('Maximum number of sunspots: %d\n', max(sunspot_count));
end

%% Function to train machine learning model for flare prediction
function trainFlareModel(data, model_file)
    if isempty(data)
        fprintf('No data available for training.\n');
        return;
    end
    
    fprintf('Training machine learning model for flare prediction...\n');
    
    % Extract features and labels from data
    num_samples = length(data);
    
    % Check if we have enough data
    if num_samples < 5
        fprintf('Not enough data for training. Need at least 5 samples.\n');
        return;
    end
    
    % Extract feature names
    feature_names = fieldnames(data(1).features);
    num_features = length(feature_names);
    
    % Create feature matrix and label vector
    X = zeros(num_samples, num_features);
    y = zeros(num_samples, 1);
    
    for i = 1:num_samples
        for j = 1:num_features
            X(i, j) = data(i).features.(feature_names{j});
        end
        y(i) = data(i).flare_class;
    end
    
    % Display distribution of flare classes
    class_names = {'None', 'C-class', 'M-class', 'X-class'};
    class_counts = histcounts(y, 0:4);
    
    fprintf('\nTraining data distribution:\n');
    for i = 1:4
        fprintf('%s: %d samples (%.1f%%)\n', class_names{i}, class_counts(i), ...
                class_counts(i)/sum(class_counts)*100);
    end
    
    % Create figure for training process visualization
    figure('Name', 'ML Model Training', 'Position', [200, 200, 1000, 800]);
    
    % Feature visualization with PCA
    subplot(2, 2, 1);
    if exist('pca', 'file')
        [~, score] = pca(X);
        scatter(score(:,1), score(:,2), 50, y, 'filled');
        colormap(jet);
        colorbar('TickLabels', class_names);
        title('PCA of Features', 'FontSize', 12);
        xlabel('Principal Component 1');
        ylabel('Principal Component 2');
    else
        % Alternative if PCA is not available
        scatter(X(:,1), X(:,2), 50, y, 'filled');
        colormap(jet);
        colorbar('TickLabels', class_names);
        title('Feature Space Visualization', 'FontSize', 12);
        xlabel(strrep(feature_names{1}, '_', ' '));
        ylabel(strrep(feature_names{2}, '_', ' '));
    end
    
    % Feature importance plot (we'll calculate this after training)
    subplot(2, 2, 2);
    
    % Split data for training and validation
    cv = cvpartition(num_samples, 'KFold', 5);
    idx_train = training(cv, 1);
    idx_test = test(cv, 1);
    
    X_train = X(idx_train, :);
    y_train = y(idx_train);
    X_test = X(idx_test, :);
    y_test = y(idx_test);
    
    % Try different ML models
    models = {'Tree', 'SVM', 'KNN', 'Ensemble'};
    accuracies = zeros(length(models), 1);
    
    % Store best model and its accuracy
    best_model = [];
    best_accuracy = 0;
    best_model_name = '';
    
    for i = 1:length(models)
        fprintf('Training %s model...\n', models{i});
        
        try
            switch models{i}
                case 'Tree'
                    mdl = fitctree(X_train, y_train);
                case 'SVM'
                    mdl = fitcecoc(X_train, y_train);
                case 'KNN'
                    mdl = fitcknn(X_train, y_train, 'NumNeighbors', 5);
                case 'Ensemble'
                    mdl = fitcensemble(X_train, y_train, 'Method', 'Bag');
            end
            
            % Predict on test set
            y_pred = predict(mdl, X_test);
            
            % Calculate accuracy
            accuracy = sum(y_pred == y_test) / length(y_test);
            accuracies(i) = accuracy;
            
            fprintf('%s model accuracy: %.2f%%\n', models{i}, accuracy * 100);
            
            % Check if this is the best model
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                best_model = mdl;
                best_model_name = models{i};
            end
            
        catch e
            fprintf('Error training %s model: %s\n', models{i}, e.message);
            accuracies(i) = 0;
        end
    end
    
    % Display model comparison
    subplot(2, 2, 3);
    bar(accuracies);
    set(gca, 'XTickLabel', models);
    title('Model Comparison', 'FontSize', 12);
    xlabel('Model Type');
    ylabel('Accuracy');
    ylim([0 1]);
    
    % Calculate and display feature importance
    if ~isempty(best_model)
        fprintf('\nBest model: %s (Accuracy: %.2f%%)\n', best_model_name, best_accuracy * 100);
        
        % Different ways to get feature importance depending on model type
        importance = zeros(num_features, 1);
        
        try
            switch best_model_name
                case 'Tree'
                    importance = predictorImportance(best_model);
                case 'Ensemble'
                    importance = predictorImportance(best_model);
                otherwise
                    % For other models, use a simple feature elimination approach
                    for j = 1:num_features
                        X_temp = X_test;
                        % Randomize one feature
                        X_temp(:, j) = X_temp(randperm(size(X_temp, 1)), j);
                        y_pred_mod = predict(best_model, X_temp);
                        acc_mod = sum(y_pred_mod == y_test) / length(y_test);
                        importance(j) = best_accuracy - acc_mod;
                    end
            end
            
            % Normalize importance
            importance = importance / sum(importance);
            
            % Plot feature importance
            subplot(2, 2, 2);
            [sorted_importance, idx] = sort(importance, 'descend');
            barh(sorted_importance);
            set(gca, 'YTickLabel', strrep(feature_names(idx), '_', ' '));
            title('Feature Importance', 'FontSize', 12);
            xlabel('Normalized Importance');
            
            % Display confusion matrix
            subplot(2, 2, 4);
            y_pred_all = predict(best_model, X);
            confusionchart(y, y_pred_all, 'RowSummary', 'row-normalized', ...
                          'ColumnSummary', 'column-normalized');
            title('Confusion Matrix', 'FontSize', 12);
            
            % Save the trained model
            model.mdl = best_model;
            model.feature_names = feature_names;
            model.accuracy = best_accuracy;
            model.importance = importance;
            model.class_names = class_names;
            save(model_file, 'model');
            
            fprintf('Model saved to %s\n', model_file);
        catch e
            fprintf('Error calculating feature importance: %s\n', e.message);
        end
    else
        fprintf('No successful model was trained.\n');
    end
end

%% Function to predict solar flares using trained ML model
function predictSolarFlaresML(data, model_file)
    if isempty(data)
        fprintf('No data available for prediction.\n');
        return;
    end
    
    % Load trained model
    try
        load(model_file, 'model');
        fprintf('Loaded ML model for flare prediction.\n');
    catch
        fprintf('Error loading model file. Please train the model first.\n');
        return;
    end
    
    fprintf('Predicting solar flares using machine learning...\n');
    
    % Create figure for prediction visualization
    figure('Name', 'ML Solar Flare Prediction', 'Position', [250, 250, 1200, 800]);
    
    % Extract features from data
    num_samples = length(data);
    num_features = length(model.feature_names);
    X = zeros(num_samples, num_features);
    
    for i = 1:num_samples
        for j = 1:num_features
            X(i, j) = data(i).features.(model.feature_names{j});
        end
    end
    
    % Make predictions
    [y_pred, scores] = predict(model.mdl, X);
    
    % For class probabilities (if available, depends on the model type)
    if size(scores, 2) == length(model.class_names)
        probs = scores;
    else
        % If direct probabilities aren't available, normalize scores
        probs = scores ./ sum(scores, 2);
    end
    
    % Plot the most recent magnetogram
    subplot(2, 2, 1);
    imagesc(data(end).field_strength);
    colormap(customMagnetogramColormap());
    colorbar;
    title('Latest Magnetogram', 'FontSize', 14);
    axis image off;
    
    % Highlight active regions with predicted flare probability
    subplot(2, 2, 2);
    imagesc(data(end).field_strength);
    colormap(customMagnetogramColormap());
    axis image off;
    
    % Find active regions based on field strength
    threshold = 1000;
    strong_field = abs(data(end).field_strength) > threshold;
    se = strel('disk', 3);
    active_regions = imclose(strong_field, se);
    active_regions = imopen(active_regions, se);
    [labeled, num_regions] = bwlabel(active_regions);
    stats = regionprops(labeled, 'Area', 'Centroid', 'BoundingBox');
    
    % Overlay region markers with prediction
    hold on;
    latest_class = y_pred(end);
    for j = 1:num_regions
        if stats(j).Area > 20  % Filter out tiny regions
            x = stats(j).Centroid(1);
            y = stats(j).Centroid(2);
            
            % Different marker colors based on predicted flare class
            if latest_class == 0
                marker_color = 'g';  % Green for no flare
            elseif latest_class == 1
                marker_color = 'y';  % Yellow for C-class
            elseif latest_class == 2
                marker_color = 'm';  % Magenta for M-class
            else
                marker_color = 'r';  % Red for X-class
            end
            
            plot(x, y, [marker_color, 'o'], 'MarkerSize', 10 + sqrt(stats(j).Area)/5, ...
                'LineWidth', 2);
            
            % Add rectangle around larger active regions
            if stats(j).Area > 100
                rectangle('Position', stats(j).BoundingBox, 'EdgeColor', marker_color, ...
                         'LineWidth', 2, 'LineStyle', '--');
            end
        end
    end
    hold off;
    
    title_str = sprintf('Active Regions with %s Flare Prediction', model.class_names{latest_class+1});
    title(title_str, 'FontSize', 12);
    
    % Plot prediction probabilities
    subplot(2, 2, 3);
    
    % Get the probability for each class for the latest observation
    latest_probs = probs(end, :);
    bar(0:3, latest_probs, 'FaceColor', [0.3 0.6 0.9]);
    set(gca, 'XTickLabel', model.class_names);
    title('24-Hour Flare Probability', 'FontSize', 12);
    xlabel('Flare Class');
    ylabel('Probability');
    ylim([0 1]);
    grid on;
    
    % Highlight the predicted class
    hold on;
    bar(latest_class, latest_probs(latest_class+1), 'FaceColor', [0.9 0.3 0.3]);
    hold off;
    
    % Add text annotations
    for i = 1:length(latest_probs)
        text(i-1, latest_probs(i) + 0.05, sprintf('%.1f%%', latest_probs(i)*100), ...
            'HorizontalAlignment', 'center');
    end
    
    % Plot feature values for the latest observation
    subplot(2, 2, 4);
    
    % Get top 5 important features
    [sorted_imp, idx] = sort(model.importance, 'descend');
    top_features = idx(1:min(5, length(idx)));
    top_names = model.feature_names(top_features);
    
    % Get normalized feature values
    latest_features = X(end, :);
    % Normalize to [0,1] range for visualization
    X_min = min(X);
    X_max = max(X);
    X_range = X_max - X_min;
    X_range(X_range == 0) = 1;  % Avoid division by zero
    X_norm = (X - X_min) ./ X_range;
    
    % Plot the top features
    barh(X_norm(end, top_features));
    set(gca, 'YTickLabel', strrep(top_names, '_', ' '));
    title('Key Feature Values (Latest)', 'FontSize', 12);
    xlabel('Normalized Value');
    grid on;
    xlim([0 1]);
    
    % Add text annotations
    for i = 1:length(top_features)
        text(X_norm(end, top_features(i)) + 0.05, i, ...
            sprintf('%.2f', latest_features(top_features(i))), ...
            'VerticalAlignment', 'middle');
    end
    
    % Print prediction summary
    fprintf('\nML Flare Prediction Summary:\n');
    fprintf('-------------------------\n');
    fprintf('Prediction for %s: %s\n', datestr(data(end).date), model.class_names{latest_class+1});
    
    % Print top probabilities
    fprintf('Flare class probabilities:\n');
    for i = 1:length(model.class_names)
        fprintf('  %s: %.1f%%\n', model.class_names{i}, latest_probs(i)*100);
    end
    
    % Print top contributing features
    fprintf('\nTop contributing features:\n');
    for i = 1:min(3, length(top_features))
        feature_name = strrep(top_names{i}, '_', ' ');
        fprintf('  %s: %.2f (Importance: %.1f%%)\n', ...
            feature_name, latest_features(top_features(i)), sorted_imp(i)*100);
    end
    
    % Create a time series forecast plot
    if num_samples > 1
        figure('Name', 'ML Flare Prediction Time Series', 'Position', [300, 300, 1000, 500]);
        
        % Plot time series of prediction probabilities
        dates = [data.date];
        
        % Stack plot of probabilities over time
        area(dates, probs);
        legend(model.class_names, 'Location', 'NorthWest');
        title('Flare Probability Forecast Over Time', 'FontSize', 14);
        xlabel('Date');
        ylabel('Probability');
        grid on;
        
        % Mark actual flare observations if available
        hold on;
        actual_classes = zeros(num_samples, 1);
        for i = 1:num_samples
            actual_classes(i) = data(i).flare_class;
        end
        
        % Add markers for actual flare events
        has_flares = actual_classes > 0;
        if any(has_flares)
            class_colors = 'gywr';  % Colors for different flare classes
            for i = 1:num_samples
                if actual_classes(i) > 0
                    plot(dates(i), 1.05, ['v', class_colors(actual_classes(i))], ...
                        'MarkerFaceColor', class_colors(actual_classes(i)), ...
                        'MarkerSize', 8 + actual_classes(i)*2);
                end
            end
            
            % Add a legend for actual events
            legend([model.class_names, 'Actual Flare Events'], 'Location', 'NorthWest');
        end
        hold off;
    end
    
    % Check for any imminent flare threats
    max_prob = max(latest_probs);
    max_class = find(latest_probs == max_prob) - 1;
    
    fprintf('\nFlare threat assessment:\n');
    if max_class == 0
        fprintf('No significant flare activity expected in the next 24 hours.\n');
    elseif max_class == 1
        if max_prob > 0.7
            fprintf('HIGH probability of C-class flare activity in the next 24 hours.\n');
        else
            fprintf('Moderate probability of C-class flare activity in the next 24 hours.\n');
        end
    elseif max_class == 2
        if max_prob > 0.7
            fprintf('WARNING: HIGH probability of M-class flare activity in the next 24 hours.\n');
            fprintf('Potential for radio blackouts and minor radiation storms.\n');
        else
            fprintf('Moderate probability of M-class flare activity in the next 24 hours.\n');
        end
    else
        if max_prob > 0.5
            fprintf('SEVERE WARNING: Significant probability of X-class flare activity in the next 24 hours.\n');
            fprintf('Potential for wide-area blackouts, radiation hazards, and geomagnetic storms.\n');
        else
            fprintf('Low to moderate probability of X-class flare activity in the next 24 hours.\n');
        end
    end
end

%% Function to create a simulated magnetogram for testing
function createSimulatedMagnetogram(filename, date)
    % Size of magnetogram
    width = 1024;
    height = 1024;
    
    % Create base solar disk
    [x, y] = meshgrid(linspace(-1, 1, width), linspace(-1, 1, height));
    r = sqrt(x.^2 + y.^2);
    solar_disk = r <= 0.95;  % Solar disk with radius slightly less than 1
    
    % Fade the edge of the Sun to simulate limb darkening
    limb_darkening = (1 - r.^2).^0.5;
    limb_darkening(r > 0.95) = 0;
    
    % Create base magnetogram (neutral gray)
    magnetogram = ones(height, width) * 128;
    
    % Set background to black
    magnetogram(~solar_disk) = 0;
    
    % Convert date to usable numeric value for seed
    date_num = datenum(date);
    rng(round(date_num));  % Set random seed based on date
    
    % Add sunspots (magnetic active regions)
    num_active_regions = 5 + floor(rand * 15);  % Random number of active regions
    
    for i = 1:num_active_regions
        % Random position within solar disk
        angle = rand * 2 * pi;
        radius = rand * 0.8;  % Keep within 0.8 of the radius
        center_x = width/2 + radius * cos(angle) * width/2;
        center_y = height/2 + radius * sin(angle) * height/2;
        
        % Size and strength of active region
        size_factor = 20 + rand * 100;
        strength = (rand * 2 - 1) * 120;  % Random strength -120 to +120
                
        % Create active region
        for y = 1:height
            for x = 1:width
                dist = sqrt((x - center_x)^2 + (y - center_y)^2);
                if dist < size_factor
                    field_value = strength * exp(-dist^2 / (2 * (size_factor/2)^2));
                    
                    % Only modify within solar disk
                    if solar_disk(y, x)
                        magnetogram(y, x) = magnetogram(y, x) + field_value;
                    end
                end
            end
        end
    end
    
    % Add some bipolar active regions (more likely to produce flares)
    num_bipolar = 1 + floor(rand * 3);
    for i = 1:num_bipolar
        % Random position within solar disk
        angle = rand * 2 * pi;
        radius = 0.2 + rand * 0.6;  % Keep within reasonable radius
        
        % Create two neighboring opposite polarity regions
        center_x1 = width/2 + radius * cos(angle) * width/2;
        center_y1 = height/2 + radius * sin(angle) * height/2;
        
        % Second pole is nearby
        angle_offset = (rand * 0.4 + 0.1) * pi;  % Small angle offset
        center_x2 = center_x1 + 50 * cos(angle + angle_offset);
        center_y2 = center_y1 + 50 * sin(angle + angle_offset);
        
        % Size and strength
        size_factor = 30 + rand * 60;
        strength = 80 + rand * 60;  % Stronger fields
        
        % Create first pole (positive)
        for y = 1:height
            for x = 1:width
                dist = sqrt((x - center_x1)^2 + (y - center_y1)^2);
                if dist < size_factor
                    field_value = strength * exp(-dist^2 / (2 * (size_factor/2)^2));
                    
                    % Only modify within solar disk
                    if solar_disk(y, x)
                        magnetogram(y, x) = magnetogram(y, x) + field_value;
                    end
                end
            end
        end
        
        % Create second pole (negative)
        for y = 1:height
            for x = 1:width
                dist = sqrt((x - center_x2)^2 + (y - center_y2)^2);
                if dist < size_factor
                    field_value = -strength * exp(-dist^2 / (2 * (size_factor/2)^2));
                    
                    % Only modify within solar disk
                    if solar_disk(y, x)
                        magnetogram(y, x) = magnetogram(y, x) + field_value;
                    end
                end
            end
        end
    end
    
    % Ensure values are within valid range [0, 255]
    magnetogram = min(max(magnetogram, 0), 255);
    
    % Apply limb darkening effect to the magnetogram
    for y = 1:height
        for x = 1:width
            if solar_disk(y, x)
                dark_factor = 0.5 + 0.5 * limb_darkening(y, x);
                magnetogram(y, x) = 128 + (magnetogram(y, x) - 128) * dark_factor;
            end
        end
    end
    
    % Convert to uint8 for image saving
    magnetogram_img = uint8(magnetogram);
    
    % Convert to RGB (for visualization)
    magnetogram_rgb = zeros(height, width, 3, 'uint8');
    
    % Custom colormap for magnetogram (blue for negative field, red for positive)
    for y = 1:height
        for x = 1:width
            if ~solar_disk(y, x)
                % Background - black
                magnetogram_rgb(y, x, :) = [0, 0, 0];
            else
                val = magnetogram(y, x);
                if val < 128  % Negative field (blue)
                    intensity = 2 * (128 - val);
                    magnetogram_rgb(y, x, :) = [0, 0, min(255, intensity)];
                elseif val > 128  % Positive field (red)
                    intensity = 2 * (val - 128);
                    magnetogram_rgb(y, x, :) = [min(255, intensity), 0, 0];
                else  % Neutral - gray
                    magnetogram_rgb(y, x, :) = [128, 128, 128];
                end
            end
        end
    end
    
    % Save image
    imwrite(magnetogram_rgb, filename);
end

%% Function to create a custom colormap for magnetograms
function cmap = customMagnetogramColormap()
    % Create a colormap: blue for negative fields, white for neutral, red for positive fields
    neg_range = linspace(0, 1, 128)';
    pos_range = linspace(0, 1, 128)';
    
    % Blue to white to red colormap
    cmap = [
        pos_range, zeros(128, 1), 1-pos_range;  % Blue to white
        ones(128, 1), zeros(128, 1), zeros(128, 1)  % White to red
    ];
end