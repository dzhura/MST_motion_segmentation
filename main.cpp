#include <iostream>
#include <cmath> // abs pow sqrt M_PI
#include <list>
#include <vector>
//#include <algorithm> // for_each
#include <utility> // pair
#include <fstream> // C++ style file reading
#include <stdio.h> // C-style file reading
#include <stdlib.h> // rand_r
#include <cassert>
#include <unordered_map> // for set -> colour

#include <libgen.h> // dirname

#include <errno.h>	

#include <sys/stat.h> // mkdir
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <opencv2/imgproc/imgproc.hpp> // split

#include <boost/property_map/vector_property_map.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/dynamic_bitset.hpp>

#define EPS 2.7182818

const bool verbose=1;

struct edge_t
{
	int _u, _v;
	double _weight;

	edge_t( int u,  int v, double weight):
		_u(u), _v(v), _weight(weight) { }

	bool operator < (const edge_t & edge) {
		return this->_weight < edge._weight;
	}
};

cv::Vec2f approx_flow_vector(double x, double y, const cv::Mat & opt_flow);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);

cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
	if( argc != 3 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1\tPath to list of optical flows" << std::endl;
		std::cout << "2\t(0,inf)\t\tthi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "3\tname of output folder" << std::endl;
		return 1;
	}

	//// Input read, check and pre processings
	std::ifstream infile(argv[1]);
	if(!infile.is_open()) {
		std::cout << "Could not open " << argv[1] << std::endl;
		return 1;
	}

	int amount_of_frames,  amount_of_frames_for_seg, amount_of_frames_to_skip;
	infile >> amount_of_frames >> amount_of_frames_for_seg >> amount_of_frames_to_skip;

	if(0 > amount_of_frames) {
		std::cout << "Negative amount of frames";
		return 1;
	}
	if(0 > amount_of_frames_to_skip) {
		std::cout << "Negative amount of frames to be skipped";
		return 1;
	}
	if(0 > amount_of_frames_for_seg) {
		std::cout << "Negative amount of frames to be segmented";
		return 1;
	}
	if(amount_of_frames-1 <= amount_of_frames_to_skip + amount_of_frames_for_seg) {
		std::cout << "Too many frames to be segmented and/or skipped";
		return 1;
	}

	std::string dname( dirname(argv[1]) );

	// Skip forward optical flows
	for(int i=0; i<amount_of_frames_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	// Read forward optical flows
	std::vector<cv::Mat> forward_flows(amount_of_frames_for_seg);
	for(size_t i=0; i<forward_flows.size(); ++i) {
		std::string opt_flow_basename;
		infile >> opt_flow_basename;
		std::string opt_flow_name = dname + '/' + opt_flow_basename;

		if( !read_opticalFlow(opt_flow_name, forward_flows[i])) {
			std::cout << "Could not read optical flow: " << opt_flow_name << std::endl;
			return 1;
		}
		if(forward_flows[i].size() != forward_flows[0].size()) {
			std::cout << "Size of optical flow " << opt_flow_name << " is different from size of the first optical flow" << std::endl;
			return 1;
		}
	}

	// Skips the rest of forward optical flows
	for(int i=amount_of_frames_to_skip + amount_of_frames_for_seg; i<amount_of_frames-1; ++i) {
		std::string dummy;
		infile >> dummy;
	}
	
	// Skip backward optical flows
	for(int i=0; i<amount_of_frames_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	// Read backward optical flows
	std::vector<cv::Mat> backward_flows(amount_of_frames_for_seg+1); // backward flows are shifted towards: 0 element is empty
	for(size_t i=1; i<backward_flows.size(); ++i) {
		std::string opt_flow_basename;
		infile >> opt_flow_basename;
		std::string opt_flow_name = dname + '/' + opt_flow_basename;

		if( !read_opticalFlow(opt_flow_name, backward_flows[i])) {
			std::cout << "Could not read optical flow: " << opt_flow_name << std::endl;
			return 1;
		}
		if(backward_flows[i].size() != forward_flows[0].size()) {
			std::cout << "Size of optical flow " << opt_flow_name << " is different from size of the first optical flow" << std::endl;
			return 1;
		}
	}

	// Finish
	infile.close();

	double thi = atof(argv[2]);
	assert( 0 <= thi );

	std::string base_output_filename(argv[3]);
	if(mkdir(base_output_filename.c_str(), 0755) == -1) {
		std::cout << "Failed to create " << base_output_filename << " directory: " << strerror(errno) << std::endl;
		return -1;
	}

	std::cout << "Amount of frames to be segmented: " << amount_of_frames_for_seg << std::endl;

	cv::Size flow_size = forward_flows[0].size();
	std::cout << "Frame size: " << flow_size << std::endl; // Flow size is equal to frame size

	//// Build a spatio-temporal graph
	// Vertices
	std::list< int > vertices_indexes;
	for(int t=0; t<amount_of_frames_for_seg; ++t) {
		for(int y=0; y<flow_size.height; ++y) {
			for(int x=0; x<flow_size.width; ++x) {
				vertices_indexes.push_back(x + y*flow_size.width + t*flow_size.width*flow_size.height);
			}
		}
       	}
	if(verbose) {
		std::cout << "Amount of vertices:\t" << vertices_indexes.size() << std::endl;
	}

	// Edges
	std::list< edge_t > edges;
	for(auto v_index=vertices_indexes.begin(); v_index != vertices_indexes.end(); ++v_index) {
		size_t frame_num = *v_index / (flow_size.width*flow_size.height);
		cv::Point v_cord;
		v_cord.x = *v_index % flow_size.width;
		v_cord.y = ((*v_index - v_cord.x)/flow_size.width) % flow_size.height;

		cv::Vec2f v_fflow_vec = forward_flows[frame_num].at<cv::Vec2f>(v_cord);

		// Build edges between four closest neightbours: top, left, right and bottom
		// More neightbours are redundant
		// Current vertex is already linked with left and top neigthbours
		std::list<cv::Point> neightbours_cord;
		if(v_cord.x+1 < flow_size.width) {
			neightbours_cord.emplace_back(v_cord.x+1, v_cord.y); // Neightbour on the right
		}
		if(v_cord.y+1 < flow_size.height) {
			neightbours_cord.emplace_back(v_cord.x, v_cord.y+1); // neightbour on the bottom
		}

		// Compute similarity
		for(auto n = neightbours_cord.begin(); n != neightbours_cord.end(); ++n) {
			int neightbour_index = n->x + n->y*flow_size.width + frame_num*flow_size.width*flow_size.height;

			double motion_similarity = cv::norm(forward_flows[frame_num].at<cv::Vec2f>(*n) - v_fflow_vec, cv::NORM_L2); // TODO normalization from Brox ECCV`10
			double geometric_similarity = sqrt(pow(n->x - v_cord.x,2) + pow(n->y - v_cord.y,2));
			// TODO double flow_error = ...

			edges.emplace_back(*v_index, neightbour_index, geometric_similarity*motion_similarity);
		}

		// Track this vertext to the next frame and give similarity equal to error of flow estimation
		// TODO Get backward optical flow
		cv::Point tracked_v_cord;
		tracked_v_cord.x = v_cord.x + cvRound(v_fflow_vec[0]);
		tracked_v_cord.y = v_cord.y + cvRound(v_fflow_vec[1]);

		if( tracked_v_cord.x < 0 || flow_size.width <= tracked_v_cord.x ||
			tracked_v_cord.y < 0 || flow_size.height <= tracked_v_cord.y ||
			frame_num == forward_flows.size()-1 ) {
			continue; // Tracked vertex is outside of the video sequence
		}
		else {
			// Note: we are in the next frame
			int tracked_v_index = tracked_v_cord.x + tracked_v_cord.y*flow_size.width + (frame_num+1)*flow_size.width*flow_size.height;

			//double motion_similarity = cv::norm(forward_flows[frame_num+1].at<cv::Vec2f>(tracked_v_cord) - v_fflow_vec, cv::NORM_L2);
			cv::Vec2f aprox_bflow = approx_flow_vector(v_cord.x + v_fflow_vec[0], v_cord.y + v_fflow_vec[1], backward_flows[frame_num+1]);
			if(isnan(aprox_bflow[0]) || isnan(aprox_bflow[1])) {
				continue;
			}
			double flow_error = cv::norm(v_fflow_vec + aprox_bflow);

			edges.emplace_back(*v_index, tracked_v_index, flow_error);
		}
	}
	if(verbose) {
		std::cout << "Amount of edges:\t" << edges.size() << std::endl;
	}

	//// Segment the spatio-temporal graph
	// Segmentation by Kruskal`s minimum spanning tree
	// flow vectors form the same set have similar orientation

	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map<int> parent_t;

	rank_t rank_map(vertices_indexes.size());
	parent_t parent_map(vertices_indexes.size());

	boost::disjoint_sets<rank_t, parent_t > dsets(rank_map, parent_map);
	for(auto vertex=vertices_indexes.begin(); vertex != vertices_indexes.end(); ++vertex) {
		dsets.make_set(*vertex);
	}

	std::vector<double> mst_max_weight(vertices_indexes.size(), 0);
	std::vector<double> mst_size(vertices_indexes.size(), 1);

	edges.sort();

	for(auto edge = edges.begin(); edge != edges.end(); ++edge) {
		int parent_u = dsets.find_set(edge->_u);
		int parent_v = dsets.find_set(edge->_v);
		
		if( parent_u != parent_v &&
			edge->_weight <= std::min(
						mst_max_weight[parent_u] + thi / mst_size[parent_u], // Similarity criteria
						mst_max_weight[parent_v] + thi / mst_size[parent_v])
					) {
			dsets.link(parent_u, parent_v); // Equivalent to union(u, v)

			int parent = dsets.find_set(parent_u);
			mst_max_weight[parent] = edge->_weight; // Since edges are sorted in ascend order
			mst_size[parent] = mst_size[parent_u] + mst_size[parent_v];
		}	
	}
	dsets.compress_sets(vertices_indexes.begin(), vertices_indexes.end());

	int amount_of_segments = dsets.count_sets(vertices_indexes.begin(), vertices_indexes.end());
	if(verbose) {
		std::cout << "Amount of segments:\t" << amount_of_segments << std::endl;
	}

	//// 
	std::unordered_map<int, int> segment2colour_index(amount_of_segments);
	int colour_index = 0;
	for( auto v = vertices_indexes.begin(); v != vertices_indexes.end(); ++v) {
		int segment_rep = dsets.find_set(*v);
		auto iter = segment2colour_index.find(segment_rep);
		if(iter == segment2colour_index.end()) {
			segment2colour_index.insert( std::pair<int, int>(segment_rep, colour_index));
			colour_index++;
		}
	}


	//// Generating colour codes
	if(amount_of_segments >= pow(256,3)) {
		if(verbose) {
			std::cout << "Too many segments!" << std::endl;
			std::cout << "return 1;" << std::endl;
		}
		return 1;
	}
	std::vector<int> random_numbers;
	unsigned int seed = rand() % 1000000;
	randomPermuteRange(/*amount of numbers*/pow(256,3), random_numbers, &seed); 

	//// Output
	std::vector<cv::Mat> output(forward_flows.size());
	for(size_t i=0; i<output.size(); ++i) {
		cv::Mat temp = cv::Mat::zeros(flow_size, CV_8UC3);
		temp.copyTo(output[i]);
	}

	for(auto p_v = vertices_indexes.begin(); p_v != vertices_indexes.end(); ++p_v) {
	
		int x = *p_v % flow_size.width;
		int y = ((*p_v - x)/flow_size.width) % flow_size.height;
		int t = *p_v / (flow_size.width*flow_size.height);
		int parent = dsets.find_set(*p_v);

		output[t].at<cv::Vec3b>(y, x) = colour_8UC3(random_numbers[segment2colour_index[parent]]);
	}

	for(size_t i=0; i < output.size(); ++i) {
		size_t optical_flow_number = 1 + amount_of_frames_to_skip + i;
		std::string output_filename = base_output_filename + '/' + std::to_string(optical_flow_number) + ".png";
		cv::imwrite(output_filename, output[i]);
	}

	return 0;
}

cv::Vec2f approx_flow_vector(double x, double y, const cv::Mat & opt_flow)
{
	if( x < 0 || opt_flow.size().width-1 <= x ||
		y < 0 || opt_flow.size().height-1 <= y ) {
		return cv::Vec2f(nan(""), nan(""));
	}

	int l_x = floor(x); // left
	int r_x = l_x + 1; // right
	int b_y = floor(y); // bottom
	int u_y = b_y + 1; // upper

	double dx = x - l_x;
	double dy = y - b_y;

	cv::Vec2f ul_flow = opt_flow.at<cv::Vec2f>(u_y, l_x);
	cv::Vec2f ur_flow = opt_flow.at<cv::Vec2f>(u_y, r_x);
	cv::Vec2f bl_flow = opt_flow.at<cv::Vec2f>(b_y, l_x);
	cv::Vec2f br_flow = opt_flow.at<cv::Vec2f>(b_y, r_x);
	
	double a = (1.0 - dx)*ul_flow[0] + dx*ur_flow[0];
	double b = (1.0 - dx)*bl_flow[0] + dx*br_flow[0];
	double u = (1.0 - dy)*b + dy*a;

	a = (1.0 - dy)*bl_flow[1] + dy*ul_flow[1];
	b = (1.0 - dy)*br_flow[1] + dy*ur_flow[1];
	double v = (1.0 - dx)*a + dx*b;

	return cv::Vec2f(u, v);
}

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed)
{
	vec.clear();
	vec.resize(n);

	vec[0]=0;
	for(int i=1, j=0; i<n; ++i) {
		j = rand_r(seed) % (i+1);
		vec[i] = vec[j];
		vec[j] = i;
	}
}

cv::Vec3b colour_8UC3(int index)
{
	unsigned char r = (unsigned char) ((index>>16) % 256);
	unsigned char g = (unsigned char) ((index>>8) % 256);
	unsigned char b = (unsigned char) (index % 256);

	return cv::Vec3b(b, g, r);

}

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out)
{
	if(opticalflow_filename.empty()) {
		return false;
	}

	FILE * stream = fopen(opticalflow_filename.c_str(), "rb");
	if(stream == 0) {
		//std::cout << "Could not open " << opticalflow_filename << std::endl;
		return false;
	}

	int width, height;
	float tag;

	if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
	       	(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
	       	(int)fread(&height, sizeof(int),   1, stream) != 1) {
		//std::cout << "Problem reading file " << opticalflow_filename << std::endl;
		return false;
	}

	if (tag != 202021.25) { // simple test for correct endian-ness
		//std::cout << "Wrong tag (possibly due to big-endian machine?)" << std::endl;
		return false;
	}

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999) {
		//std::cout << "Illegal width " << opticalflow_filename << std::endl;
		return false;
	}

	if (height < 1 || height > 99999) {
		//std::cout << "Illegal height " << opticalflow_filename << std::endl;
		return false;
	}

	cv::Size flow_size(width, height);

	size_t flo_data_size = 2 * flow_size.area() * sizeof(float);
	unsigned char * flo_data = new unsigned char[flo_data_size];

	size_t ret_code = fread(flo_data, sizeof(float), 2*flow_size.area(), stream);

	if(ret_code != 2*(size_t)flow_size.area()) {
		if(feof(stream)) {
			//std::cout << "Error reading " << opticalflow_filename << ": unexpected end of the file" << std::endl;
			return false;
		}
		else if(ferror(stream)) {
			//std::cout << "Error reading " << opticalflow_filename << std::endl;
			return false;
		}
	}

	if (fgetc(stream) != EOF) {
		//std::cout << "File is too long" << opticalflow_filename << std::endl;
		return false;
	}

	fclose(stream);

	cv::Mat(flow_size, CV_32FC2, flo_data).copyTo(out);

	delete[] flo_data;

	return true;
}
