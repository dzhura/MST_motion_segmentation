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

// Returns nan is lo or ro are zero vectros
// TODO Use homogeneous coordinates to prevent divizion by zero
float cosine_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro);

float l2_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
	if( argc != 5 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1\tPath to list of optical flows. First row: <amount of optical flows> <optical flows to skip> <optical flows for segmetation>. Second row: <path to optical flow>, third row: <path to optical flow>, etc" << std::endl;
		std::cout << "2\t(0,inf)\t\tthi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "3\t2*n+1, n=1,2,3...\tsize of spatial window, where all elements are linked with the central" << std::endl;
		std::cout << "4\t2*n+1, n=1,2,3...\tsize of temporal window, where all elements are linked with the central" << std::endl;
		std::cout << "5\tbase name of output file" << std::endl;
		return 1;
	}

	//// Input read, check and pre processings
	std::ifstream infile(argv[1]);
	if(!infile.is_open()) {
		std::cout << "Could not open " << argv[1] << std::endl;
		return 1;
	}
	std::string dname( dirname(argv[1]) );

	int amount_of_optical_flows=0,  amount_of_optical_flows_to_skip=0, amount_of_optical_flows_for_seg=0;
	infile >> amount_of_optical_flows >> amount_of_optical_flows_to_skip >> amount_of_optical_flows_for_seg;

	assert(0 < amount_of_optical_flows);
	assert(-1 < amount_of_optical_flows_to_skip);
	assert(0 < amount_of_optical_flows_for_seg);
	assert(amount_of_optical_flows_to_skip + amount_of_optical_flows_for_seg <= amount_of_optical_flows);

	// Skip
	for(int i=0; i<amount_of_optical_flows_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	std::vector<cv::Mat> flows(amount_of_optical_flows_for_seg);
	for(size_t i=0; i<flows.size(); ++i) {
		std::string opt_flow_basename;
		infile >> opt_flow_basename;
		std::string opt_flow_name = dname + '/' + opt_flow_basename;

		if( !read_opticalFlow(opt_flow_name, flows[i])) {
			std::cout << "Could not read optical flow: " << opt_flow_name << std::endl;
			return 1;
		}
		if(flows[i].size() != flows[0].size()) {
			std::cout << "Size of optical flow " << opt_flow_name << " is different from size of the first optical flow" << std::endl;
			return 1;
		}
	}
	infile.close();

	cv::Size flow_size = flows[0].size();

	double thi = atof(argv[2]);
	assert( 0 < thi );

	int spatial_window_size = atoi(argv[3]);
	assert( 3 <= spatial_window_size && spatial_window_size < cv::min(flows[0].cols, flows[0].rows) && spatial_window_size%2 == 1 );

	size_t temporal_window_size = atoi(argv[4]);
	assert( 1 <= temporal_window_size && temporal_window_size <= flows.size() && temporal_window_size%2 == 1 );

	std::string base_output_filename(argv[5]);

	//// Build an spatio-temporal graph
	std::list< int > vertices;
	for(size_t t=0; t<flows.size(); ++t) {
	for(int y=0; y<flow_size.height; ++y) {
	for(int x=0; x<flow_size.width; ++x) {

		int u_index = x + y*flow_size.width + t*flow_size.width*flow_size.height;
		vertices.push_back(u_index);
	} } }
	if(verbose) {
		std::cout << "Amount of vertices:\t" << vertices.size() << std::endl;
	}

	std::list< edge_t > edges;
	for(auto p_u=vertices.begin(); p_u != vertices.end(); ++p_u) {
		
		int u_x = *p_u % flow_size.width;
		int u_y = ((*p_u - u_x)/flow_size.width) % flow_size.height;
		int u_t = *p_u / (flow_size.width*flow_size.height);

		for( int y = u_y+1; y <= cv::min(flow_size.height-1, u_y+spatial_window_size/2); ++y) {

			int v_index = u_x + y*flow_size.width + u_t*flow_size.width*flow_size.height;
			edges.emplace_back(*p_u, v_index, l2_similarity(flows[u_t].at<cv::Vec2f>(u_y, u_x), flows[u_t].at<cv::Vec2f>(y, u_x)));
		}

		for( int x = u_x+1; x <= cv::min(flow_size.width-1, u_x+spatial_window_size/2); ++x) {

			int v_index = x + u_y*flow_size.width + u_t*flow_size.width*flow_size.height;
			edges.emplace_back(*p_u, v_index, l2_similarity(flows[u_t].at<cv::Vec2f>(u_y, u_x), flows[u_t].at<cv::Vec2f>(u_y, x)));
		}

		for( int y = u_y+1; y <= cv::min(flow_size.height-1, u_y+spatial_window_size/2); ++y) {
		for( int x = u_x+1; x <= cv::min(flow_size.width-1, u_x+spatial_window_size/2); ++x) {

			int v_index = x + y*flow_size.width + u_t*flow_size.width*flow_size.height;
			edges.emplace_back(*p_u, v_index, l2_similarity(flows[u_t].at<cv::Vec2f>(u_y, u_x), flows[u_t].at<cv::Vec2f>(y, x)));
		} }

		cv::Vec2f flow_vec = flows[u_t].at<cv::Vec2f>(u_y, u_x);
		float x = u_x + flow_vec[0];
		float y = u_y + flow_vec[1];
		for( size_t t = u_t+1; t <= cv::min(flows.size()-1, u_t+temporal_window_size/2); ++t) {

			int rounded_x = cvRound(x);
			if( rounded_x < 0 || flow_size.width <= rounded_x) {
				continue;
			}
			int rounded_y = cvRound(y);
			if( rounded_y < 0 || flow_size.height <= rounded_y) {
				continue;
			}

			int v_index = rounded_x + rounded_y*flow_size.width + t*flow_size.width*flow_size.height;
			edges.emplace_back(*p_u, v_index, l2_similarity(flows[u_t].at<cv::Vec2f>(u_y, u_x), flows[t].at<cv::Vec2f>(rounded_y, rounded_x)));

			flow_vec = flows[t].at<cv::Vec2f>(rounded_y, rounded_x);
			x += flow_vec[0];
			y += flow_vec[1];
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

	rank_t rank_map(vertices.size());
	parent_t parent_map(vertices.size());

	boost::disjoint_sets<rank_t, parent_t > dsets(rank_map, parent_map);
	for(auto vertex=vertices.begin(); vertex != vertices.end(); ++vertex) {
		dsets.make_set(*vertex);
	}

	std::vector<double> mst_max_weight(vertices.size(), 0);
	std::vector<double> mst_size(vertices.size(), 1);

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
			mst_max_weight[parent] = edge->_weight; // Edges are sorted
			mst_size[parent] = mst_size[parent_u] + mst_size[parent_v];
		}	
	}
	dsets.compress_sets(vertices.begin(), vertices.end());

	int amount_of_segments = dsets.count_sets(vertices.begin(), vertices.end());
	if(verbose) {
		std::cout << "Amount of segments:\t" << amount_of_segments << std::endl;
	}

	//// 
	std::unordered_map<int, int> segment2colour_index(amount_of_segments);
	int colour_index = 0;
	for( auto v = vertices.begin(); v != vertices.end(); ++v) {
		int segment_rep = dsets.find_set(*v);
		auto iter = segment2colour_index.find(segment_rep);
		if(iter == segment2colour_index.end()) {
			segment2colour_index.insert( std::pair<int, int>(segment_rep, colour_index));
			colour_index++;
		}
	}


	//// Generating colour codes
	std::vector<int> random_numbers;
	unsigned int seed = rand() % 1000000;
	randomPermuteRange(pow(256,3), random_numbers, &seed); 

	//// Output
	std::vector<cv::Mat> output(flows.size());
	for(size_t i=0; i<output.size(); ++i) {
		cv::Mat temp = cv::Mat::zeros(flow_size, CV_8UC3);
		temp.copyTo(output[i]);
	}

	for(auto p_v = vertices.begin(); p_v != vertices.end(); ++p_v) {
	
		int x = *p_v % flow_size.width;
		int y = ((*p_v - x)/flow_size.width) % flow_size.height;
		int t = *p_v / (flow_size.width*flow_size.height);
		int parent = dsets.find_set(*p_v);

		output[t].at<cv::Vec3b>(y, x) = colour_8UC3(random_numbers[segment2colour_index[parent]]);
	}

	for(size_t i=0; i < output.size(); ++i) {
		std::string output_filename = base_output_filename + '_' + std::to_string(i) + ".png";
		cv::imwrite(output_filename, output[i]);
	}

	return 0;
}

float cosine_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro)
{
	float similarity  = (lo[0]*ro[0] + lo[1]*ro[1]) / (cv::norm(lo) * cv::norm(ro));

	if(isnan(similarity)) {
		return similarity;
	}
	return (similarity > 0)? 1 - similarity : 1;
}
float l2_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro)
{
	return cv::norm(lo - ro, cv::NORM_L2);
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
