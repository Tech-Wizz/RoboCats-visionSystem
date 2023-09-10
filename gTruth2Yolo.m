function gTruth2Yolo(gTruth,destination_directory)

    % destination = './temp';
    mkdir(destination_directory);

    videoPath = gTruth.DataSource.Source;
    videoObj = VideoReader(videoPath);

    % Extract Properties
    width = videoObj.Width;
    height = videoObj.Height;
    numFrames = videoObj.NumFrames;
    labelNames = gTruth.LabelData(1,:).Properties.VariableNames;
    numLabelTypes = length(labelNames);
    

    for i = 1:numFrames
        % Getting video frames and saving every video frame as a .jpg image
        curFrame = readFrame(videoObj);
        filename = [destination_directory '/frame_' num2str(i)];
        vidfilename = [filename '.jpg'];
        imwrite(curFrame,vidfilename,'JPEG');
        
        % Taking in the Class labels
        labeltable = gTruth.LabelData(i,:);
        labelmat = table2array(labeltable);
        full_temp = [];
        % for every label in the frame
        for j = 1:numLabelTypes
            label_temp = labelmat{j};
            num_instances = size(label_temp,1);
            label_temp = [(j-1)*ones(num_instances,1), label_temp];
            
            full_temp = [full_temp; label_temp];
        end
        full_temp(:,1) = round(full_temp(:,1));
        full_temp(:,2) = full_temp(:,2)/width;          % Determine left edge as % of pixels
        full_temp(:,3) = full_temp(:,3)/height;         % Determine right edge as % of pixels
        full_temp(:,4) = full_temp(:,4)/width;          % Determine width as % of pixels
        full_temp(:,5) = full_temp(:,5)/height;         % Determine height as % of pixels
        full_temp(:,4) = full_temp(:,2)+full_temp(:,4); % Convert width to right edge
        full_temp(:,5) = full_temp(:,3)+full_temp(:,5); % Convert height to bottom edge
        
        for row = 1:size(full_temp,1)
            for col = 2:5
                if(full_temp(row,col) > 1)
                    full_temp(row,col) = 1;
                elseif(full_temp(row,col) < 0)
                    full_temp(row,col) = 0;
                end
            end
        end

        text_row = num2str(full_temp);

        numRows = size(text_row,1);


        labelfilename = [filename '.txt'];
        % fid = fopen(labelfilename,'wt');
        % write every row of strings into the txt file, if there is a file
        for j = 1:numRows
            toWrite = text_row(j,:);
            toWrite = regexprep(toWrite,'\s+',' ');
            if j == 1
                writelines(toWrite,labelfilename,WriteMode="overwrite");
            else
                writelines(toWrite,labelfilename,WriteMode="append");
            end
        end

    end

end