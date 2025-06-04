
Sports Videos in the Wild (SVW) dataset is accompanied with two CSV files. "SVW.csv" includes information about all the videos in the dataset, including sports genre, video length and resolution, whether bounding boxes for each video is provided or not, and three train/test split for genre categorization performance evaluation. 

"BoundingBoxes.csv" includes action annotation and bounding boxes for a large portion of the videos in SVW. For each action, 2 or 3 bounding boxes at start/end or start/mid/end frames of the action are specified (depending on complexity of the action movement). Each box is specified by 4 numbers ranging between 0 and 1, specifying x- and y- coordinate of the top left corner and width and height of the box respectively.  

The Matlab code "overlayBBsCutActions.m" overlays "Bounding Boxes" in SVW dataset over the input videos and also segments out the actions defined in bounding boxes for each video and writes the result in an output video. 


Citation:
@inproceedings{Safdarnejad2015,
author = {Safdarnejad, S. Morteza and Liu, Xiaoming and Udpa, Lalita and Andrus, Brooks and Wood, John and Craven, Dean},
title = {Sports Videos in the Wild (SVW): A Video Dataset for Sports Analysis},
booktitle={Proc. of the IEEE International Conference on Automatic Face and Gesture Recognition},
year={2015},
organization={IEEE}
}




