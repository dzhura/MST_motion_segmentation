Segmentation of a video based on MST clustering of optical flows
The MST clustering was firstly introduced in [1]

Required: Linux, OpenCV 2.4, boost::disjoint_sets and g++ with support of C++11

OpticalFlowSegmentation usage:
1	Path to list of optical flows
2	(0,inf)		thi additional threshold; larger thi results in larger segments; 2.5 is a good default value
3	name of output folder

Given a number of frames N. Segment M frames based on provided forward and backward optical flows starting from frame S. S starts from 0
Output: frames from S-th to (M+S)-th <= N will be segmented

Format of the list of optical flows:
N M S
path toforward optical_flow_1
path toforward optical_flow_2
path toforward optical_flow_3
...
path toforward optical_flow_N-1
path tobackward optical_flow_2
path tobackward optical_flow_3
...
path tobackward optical_flow_N

For a demonstration, example is provided. Around 1gb of RAM is required. Part of MPII Cooking datasets[2] was used as example video. The optical flows were extracted by Deepflow[3]

References:
[1] P. Felzenszwalb, D. Huttenlocher: Efficient Graph-Based Image Segmentation. IJCV 59(2) (September 2004)
[2] A Database for Fine Grained Activity Detection of Cooking Activities, M. Rohrbach, S. Amin, M. Andriluka and B. Schiele, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June, (2012) 
[3] DeepFlow: Large displacement optical flow with deep matching Philippe Weinzaepfel, Jerome Revaud, Zaid Harchaoui and Cordelia Schmid, Proc. ICCV'13, December, 2013.
