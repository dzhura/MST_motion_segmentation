Segmentation of a video based on MST clustering of optical flows
The MST clustering was firstly introduced in [1]

OpticalFlowSegmentation usage:
1	Path to list of optical flows
2	(0,inf)		thi additional threshold; larger thi results in larger segments
3	name of output folder

Given a number of frames N. Segment M frames based on provided forward and backward optical flows starting from frame S
Output: frames S to M+S < N will be segmented

Format of the list of optical flows:
N M S
forward optical_flow_1
forward optical_flow_2
forward optical_flow_3
...
forward optical_flow_N-1
backward optical_flow_2
backward optical_flow_3
...
backward optical_flow_N

[1] P. Felzenszwalb, D. Huttenlocher: Efficient Graph-Based Image Segmentation. IJCV 59(2) (September 2004)
