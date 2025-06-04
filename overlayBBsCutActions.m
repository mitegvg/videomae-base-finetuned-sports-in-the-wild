%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Developed in Computer Vision Lab at Michigan State University%%%%%%%
%%%%%%%%%%http://www.cse.msu.edu/~liuxm/sportsVideo/index.html%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code overlays "Bounding Boxes" in SVW dataset over the input videos
% and also segments out the actions defined in BB for each video and writes
% the result in an output video. 

labelPath = './';                           % Path to the bounding box labels
filename = 'BoundingBoxes.csv'              % Name of the label file

inputVideoPath = 'C:/SVW/';                 % Path to SVW videos
outputVideoPath = './OverlaidActions/';     % Write path for videos with overlaid BBs
outputActionPath = './CutOutActions/';      % Write path for action videos cut out from SVW videos

minVideoLength = 45;                        % Specify min video length in terms of frames
spatialExpansion = 0.05;                    % Specify a spatial expansion around the bounding box for the video cut out from the bounding box (video width and height range between 0 and 1)

[~,~,txt] = xlsread([labelPath,'/',filename]);

mkdir(outputVideoPath)
mkdir(outputActionPath)
prevProcessedFile = '';

for i=2:size(txt,1)
    fileName = txt{i,1};
    fileName = fileName(1:strfind(fileName,'.')-1); 
    pureFileName = fileName(strfind(fileName,'/')+1:end);
    folderName = txt{i,1};    
    if (~isnan(txt{i,5})) % If bounding box is defined
        if (~strcmp(prevProcessedFile, folderName))
            folderName = folderName(1:strfind(folderName,'/')-1);
            Obj = VideoReader([inputVideoPath,'/',txt{i,1}]);
            w = Obj.Width; h = Obj.Height;
            numActions = 0;
            try % Find out how many actions are defined in the current video
                while strcmp(txt{i+numActions,1},txt{i,1})
                    numActions = numActions + 1;
                end
            end

            mkdir([outputVideoPath,'/',folderName]);

            writerObj = VideoWriter([outputVideoPath,'/',fileName,'.avi'],'MPEG-4');
            writerObj.FrameRate = Obj.FrameRate;
            open(writerObj);
            for j=0:numActions-1
                actionStartFrame = txt{i+j,2}; 
                actionEndFrame = txt{i+j,3};
                if (actionStartFrame == 0)
                    actionStartFrame = 1;
                    actionEndFrame = Obj.NumberOfFrames;
                end
                actionName = txt{i+j,4};
                mkdir([outputActionPath,'/',actionName]);  
                writerActObj = VideoWriter([outputActionPath,'/',actionName,'/',pureFileName,'_action_',num2str(j+1),'.avi'],'MPEG-4');                
                
                % Get the bounding box from labels
                B1(1) = max(0,txt{i+j,5}); B1(2) = max(0,txt{i+j,6}); B1(3) = min(1,txt{i+j,7}); B1(4) = min(1,txt{i+j,8});
                B2(1) = max(0,txt{i+j,9}); B2(2) = max(0,txt{i+j,10}); B2(3) = min(1,txt{i+j,11}); B2(4) = min(1,txt{i+j,12});

                if (j==0)
                    startR = 1;
                else
                    startR = txt{i+j-1,3}+1;
                end

                if (j==(numActions-1))
                    endR = Obj.NumberOfFrames;
                else
                    endR = actionEndFrame;
                end
                
                % Overlay bounding box
                for k=startR:endR            
                    inFrame = read(Obj,k);
                    if (k>=actionStartFrame & k<=actionEndFrame)
                        % if there is a middle BB, use that, otherwise
                        % only use start and end BB
                        if (size(txt,2)>13)
                            if(~isnan(txt{i+j,18}))
                                midBBFlag = 1;                              
                            else
                                midBBFlag = 0;
                            end;
                        else
                            midBBFlag = 0;
                        end
                        
                        % Bounding box at each frame is interpolated from
                        % start, mid, and end bounding boxes
                        if (midBBFlag)                            
                            midF = txt{i+j,14};
                            BM(1) = max(0,txt{i+j,15}); BM(2) = max(0,txt{i+j,16}); BM(3) = min(1,txt{i+j,17}); BM(4) = min(1,txt{i+j,18});
                            if (k>=actionStartFrame & k<=midF)
                                slope = (BM-B1)/(midF-actionStartFrame);
                                midB = slope.*(k-actionStartFrame)+B1;
                                midB = midB.*[w h w h];                                
                            else
                                slope = (B2-BM)/(actionEndFrame-midF);
                                midB = slope.*(k-midF)+BM;
                                midB = midB.*[w h w h];                                    
                            end
                        else                                                    
                            slope = (B2-B1)/(actionEndFrame-actionStartFrame);
                            midB = slope.*(k-actionStartFrame)+B1;
                            midB = midB.*[w h w h];                          
                        end

                        x1 = max(1,min(w-1,round(midB(1))));
                        x2 = max(1,min(w-1,round(midB(1)+midB(3))));
                        y1 = max(1,min(h-1,round(midB(2))));
                        y2 = max(1,min(h-1,round(midB(2)+midB(4))));

                        % Overlay bounding box on input frames (directly
                        % works on pixels for speed-up)
                        inFrame(y1:y1+1,x1:x2,1) = 255;inFrame(y1:y1+1,x1:x2,2) = 0;inFrame(y1:y1+1,x1:x2,3) = 0;
                        inFrame(y2:y2+1,x1:x2,1) = 255;inFrame(y2:y2+1,x1:x2,2) = 0;inFrame(y2:y2+1,x1:x2,3) = 0;
                        inFrame(y1:y2,x1:x1+1,1) = 255;inFrame(y1:y2,x1:x1+1,2) = 0;inFrame(y1:y2,x1:x1+1,3) = 0;
                        inFrame(y1:y2,x2:x2+1,1) = 255;inFrame(y1:y2,x2:x2+1,2) = 0;inFrame(y1:y2,x2:x2+1,3) = 0;                        
                    end
                    writeVideo(writerObj,inFrame);
                end
                
                % Segment out action and save in stand-alone video
                if (midBBFlag) 
                    cutOutBox = [min(min(B1(1:2),B2(1:2)),BM(1:2)),max(max(B1(1:2)+B1(3:4),B2(1:2)+B2(3:4)),BM(1:2)+BM(3:4))];
                    cutOutBox(3) = cutOutBox(3) - cutOutBox(1);
                    cutOutBox(4) = cutOutBox(4) - cutOutBox(2);                    
                else
                    cutOutBox = [min(B1(1:2),B2(1:2)),max(B1(1:2)+B1(3:4),B2(1:2)+B2(3:4))];
                    cutOutBox(3) = cutOutBox(3) - cutOutBox(1);
                    cutOutBox(4) = cutOutBox(4) - cutOutBox(2);                      
                end
                cutOutBox(1) = max(0, cutOutBox(1) - spatialExpansion);
                cutOutBox(2) = max(0, cutOutBox(2) - spatialExpansion);
                cutOutBox(3) = min(1, cutOutBox(3) + 2*spatialExpansion);
                cutOutBox(4) = min(1, cutOutBox(4) + 2*spatialExpansion);
                startRActWrite = actionStartFrame;
                endRActWrite = actionEndFrame;  
                
                % If video is too short, increase its length to
                % "minVideoLength"
                lengthDiff = minVideoLength - (actionEndFrame - actionStartFrame);
                if (lengthDiff>0)
                    startRActWrite = max(1,actionStartFrame - lengthDiff/2);
                    endRActWrite = min(Obj.NumberOfFrames, startRActWrite + minVideoLength); 
                end
                open(writerActObj);
                for k=startRActWrite:endRActWrite 
                    inFrame = read(Obj,k);
                    writeVideo(writerActObj,imcrop(inFrame, cutOutBox.*[w,h,w,h]))
                end
                close(writerActObj);
            end
            close(writerObj);    
            prevProcessedFile = txt{i,1};
        end
    end
end
