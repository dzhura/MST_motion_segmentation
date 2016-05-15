#include <iostream>
#include <cmath> // abs pow sqrt M_PI
#include <list>
#include <utility> // pair
#include <vector>
#include <fstream> // C++ style file reading
#include <stdio.h> // C-style file reading
#include <stdlib.h> // rand_r
#include <cassert>

#include <libgen.h> // dirname

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp> // split

#include <boost/property_map/vector_property_map.hpp>
#include <boost/pending/disjoint_sets.hpp>

#define EPS 2.7182818

const bool verbose=1;

const int norm_type = cv::NORM_L2;

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

double similarity(const float beta, float scale_factor, float eps=0);
double similarity(const int beta, float scale_factor, float eps=0);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
	if( argc != 9 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1\tfile list, including amount of frames, amount of frames for segmentation, frames to skip, frame paths and optical flow paths" << std::endl;
		std::cout << "2\t(0,inf)\t\tinverse weight for distance similarity" << std::endl;
		std::cout << "3\t(0,inf)\t\tinverse weight for appearence similarity" << std::endl;
		std::cout << "4\t(0,inf)\t\tinverse weight for motion similarity" << std::endl;
		std::cout << "5\t2*n-1, n=1,2,3...\tregion size for appearence similarity" << std::endl;
		std::cout << "6\t[0,inf)\t\tthi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "7\t2*n+1, n=1,2,3...\tsize of spatial window, where all elements are linked with the central" << std::endl;
		std::cout << "8\t2*n-1, n=1,2,3...\tsize of temporal window, where all elements are linked with the central" << std::endl;
		std::cout << "9\toutput file" << std::endl;
		return 1;
	}

	//// Input read, check and pre processings
	std::ifstream infile(argv[1]);
	if(!infile.is_open()) {
		std::cout << "Could not open " << argv[1] << std::endl;
		return 1;
	}
	std::string dname( dirname(argv[1]) );

	int amount_of_frames=0, amount_of_frames_for_seg = 0, frames_to_skip=0;
	infile >> amount_of_frames >> amount_of_frames_for_seg >> frames_to_skip;

	assert(0 < amount_of_frames);
	assert(0 < frames_to_skip); // Very first frame is never segmented
	assert(0 < amount_of_frames_for_seg);
	assert(amount_of_frames_for_seg + frames_to_skip < amount_of_frames); // Very last frame is never segmented

	// Skip first frames
	for(int i=0; i<frames_to_skip-1; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	// Always load first and last frames for appearence similarity computation
	std::vector<cv::Mat> frame(amount_of_frames_for_seg+2);
	for(size_t i=0; i<frame.size(); ++i) {
		std::string frame_bname;
		infile >> frame_bname;
		std::string frame_name = dname + '/' + frame_bname;

		frame[i] = cv::imread(frame_name);
		if(!frame[i].data) {
			std::cout << "Failed to read image " << frame_name << std::endl;
			return 1;
		}
		if(frame[i].depth() != CV_8U && frame[i].depth() != CV_16U) {
			std::cout << frame_name << " has inappropriate image depth" << std::endl;
			std::cout << "It should be unsigned 8-bit or unsigned 16-bit depth" << std::endl;
			return 1;
		}
		if(frame[i].size() != frame[0].size()) {
			std::cout << "Frames aren`t of the same size" << std::endl;
			return 1;
		}
	}

	// Skip last frames
	for(size_t i=0; i < amount_of_frames - (frames_to_skip-1) - frame.size(); ++i) {
		std::string dummy;
		infile >> dummy;
	}

	// Skip first opt flows
	for(int i=0; i<frames_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	std::vector<cv::Mat> flow(amount_of_frames_for_seg+2);
	for(int i=1; i<amount_of_frames_for_seg+2-1; ++i) {
		std::string opt_flow_bname;
		infile >> opt_flow_bname;
		std::string opt_flow_name = dname + '/' + opt_flow_bname;

		if( !read_opticalFlow(opt_flow_name, flow[i])) {
			std::cout << "Could not read optical flow: " << opt_flow_name << std::endl;
			return 1;
		}
		if(flow[i].size() != frame[0].size()) {
			std::cout << "Size of optical flow " << opt_flow_name << " is different from size of the original image" << std::endl;
			return 1;
		}
	}
	infile.close();

	float dist_similarity_scale = atof(argv[2]);
	if(dist_similarity_scale <= 0) {
		std::cout << "Scaling factor for distance similarity should be positive" << std::endl;
		return 1;
	}

	float appearence_similarity_scale = atof(argv[3]);
	if(appearence_similarity_scale <= 0) {
		std::cout << "Scaling factor for appearence similarity should be positive" << std::endl;
		return 1;
	}

	float motion_similarity_scale = atof(argv[4]);
	if(motion_similarity_scale <= 0) {
		std::cout << "Scaling factor for motion similarity should be positive" << std::endl;
		return 1;
	}

	int neighbourhood_size = atoi(argv[5]);
	if(!(0 < neighbourhood_size && neighbourhood_size%2 == 1)) {
		std::cout << "Neighbourhood size should be positive and odd" << std::endl;
		return 1;
	}

	double thi = atof(argv[6]);
	if( thi < 0 ) {
		std::cout << "The thi should be non-negative" << std::endl;
		return 1;
	}

	int spatial_window_size = atoi(argv[7]);
	if( !(3 <= spatial_window_size && spatial_window_size < cv::min(frame[0].cols, frame[0].rows) && spatial_window_size%2 == 1) ) {
		std::cout << "The size of special window should be odd, not less 3 and not large frame size " << spatial_window_size << std::endl;
		return 1;
	}

	int temporal_window_size = atoi(argv[8]);
	if( !(0 < temporal_window_size && temporal_window_size <= amount_of_frames_for_seg && temporal_window_size%2 == 1) ) {
		std::cout << "The size of temporal window should be odd, positive and not large amount of frames" << std::endl;
		std::cout << "arg.8: " << temporal_window_size << std::endl;
		return 1;
	}

	std::string base_output_filename(argv[9]);

	//// Preprocessing
	// Normalize and convert to gray scale
	std::vector<cv::Mat> gray_frame(frame.size());
	for(size_t i=0; i < gray_frame.size(); ++i) {
		switch(frame[i].depth()) {
			case CV_8U:
				frame[i].convertTo(frame[i], CV_32F, 1.0/255.0);
				break;
			case CV_16U:
				frame[i].convertTo(frame[i], CV_32F, 1.0/65535.0);
				break;
		}
		cv::cvtColor(frame[i], gray_frame[i], CV_RGB2GRAY);
	}

	// Compute mean and st deviation of the images
	// TODO test it
	std::vector<cv::Mat> gray_frame_mean(gray_frame.size());
	std::vector<cv::Mat> gray_frame_dev(gray_frame.size());
	for(size_t i=1; i < gray_frame.size()-1; ++i) {
		gray_frame_mean[i].create(gray_frame[0].size(), CV_32FC1);
		gray_frame_dev[i].create(gray_frame[0].size(), CV_32FC1);

		cv::Mat kernel = cv::Mat::ones(neighbourhood_size, neighbourhood_size, CV_32F) / (float)pow(neighbourhood_size, 2);

		cv::filter2D(gray_frame[i], gray_frame_mean[i], CV_32F, kernel);

		cv::subtract(gray_frame[i], gray_frame_mean[i], gray_frame_dev[i-1], cv::noArray(), CV_32F);
		cv::pow(gray_frame_dev[i], 2, gray_frame_dev[i]);
		cv::filter2D(gray_frame_dev[i], gray_frame_dev[i], CV_32F, kernel);
		cv::pow(gray_frame_dev[i], 0.5, gray_frame_dev[i]);
	}

	// Compute normalized forward/backward appearence change
	std::vector<cv::Mat> app_change(gray_frame.size());
	for(size_t i=1; i<gray_frame.size()-1; ++i) {
		cv::Mat backward_app_change(gray_frame[0].size(), CV_32FC1);
		cv::Mat forward_app_change(gray_frame[0].size(), CV_32FC1);

		cv::subtract(gray_frame[i-1], gray_frame[i], backward_app_change, cv::noArray(), CV_32F);
		cv::subtract(gray_frame[i], gray_frame[i+1], forward_app_change, cv::noArray(), CV_32F);

		backward_app_change = cv::abs(backward_app_change);
		forward_app_change = cv::abs(forward_app_change);

		// TODO Normalization

		std::vector<cv::Mat> separate_app_change(2);
		separate_app_change[0] = backward_app_change;
		separate_app_change[1] = forward_app_change;

		cv::merge(separate_app_change, app_change[i]);
	}

	// Convert optical flow to polar coordiantes and normalize
	for(size_t i=1; i < frame.size()-1; ++i) {
		float max_flow_magnitude = 0;
		float max_flow_angle = 0;
		for(auto flow_elm = flow[i].begin<cv::Vec2f>(); flow_elm != flow[i].end<cv::Vec2f>(); ++flow_elm) {
			float u = (*flow_elm)[0], v = (*flow_elm)[1];

			if( u > 1e9 || v > 1e9 ) { // if flow value is unknow
				continue;
			}
			if( u == 0 && v == 0 ) { // What TODO with zero flow vectors?
				continue;
			}

			(*flow_elm)[0] = sqrt(pow(u,2) + pow(v,2));
			(*flow_elm)[1] = (v >= 0)?  atan2(v,u) : 2*M_PI + atan2(v,u); // in (0; 2*M_PI]

			if((*flow_elm)[0] > max_flow_magnitude) {
				max_flow_magnitude = (*flow_elm)[0];
			}
			if((*flow_elm)[1] > max_flow_angle) {
				max_flow_angle = (*flow_elm)[1];
			}
		}
		for(auto flow_elm = flow[i].begin<cv::Vec2f>(); flow_elm != flow[i].end<cv::Vec2f>(); ++flow_elm) {
			if( (*flow_elm)[0] > 1e9 || (*flow_elm)[1] > 1e9 ) { // if flow value is unknow
				continue;
			}

			(*flow_elm)[0] /= max_flow_magnitude; // in [0;1]
			(*flow_elm)[1] /= max_flow_angle; // in (0;1]
		}
	}

	//// Create a graph model
	std::list< int > vertices;
	std::list< edge_t > edges;

	// Pixel indexes are shifted one frame back
	cv::Size video_resolution = frame[0].size();
	for(size_t v_t=1; v_t < frame.size()-1; ++v_t) {
	for(int v_y=0; v_y < video_resolution.height; ++v_y) {
	for(int v_x=0; v_x < video_resolution.width; ++v_x) {

		int v_index = v_x + v_y*video_resolution.width + (v_t-1)*video_resolution.area(); // skip first frame
		vertices.push_back(v_index);

		cv::Vec2f v_flow = flow[v_t].at<cv::Vec2f>(v_y, v_x);
		//cv::Vec3i v_p(v_x, v_y, v_t);
		//cv::Vec4f v_app(gray_frame_mean[v_t].at<float>(v_y, v_x), gray_frame_dev[v_t].at<float>(v_y, v_x), app_change[v_t].at<cv::Vec2f>(v_y, v_x)[0], app_change[v_t].at<cv::Vec2f>(v_y, v_x)[1]);

		for(size_t t = std::max((size_t)1, v_t - temporal_window_size/2); t < std::min(frame.size()-1, v_t+1 + temporal_window_size/2); ++t) {
		for(int y = std::max(0, v_y - spatial_window_size/2); y < std::min(video_resolution.height, v_y+1 + spatial_window_size/2); ++y) {
		for(int x = std::max(0, v_x - spatial_window_size/2); x < std::min(video_resolution.width, v_x+1 + spatial_window_size/2); ++x) {
		
			//cv::Vec3i p(x, y, t);
			//cv::Vec4f app(gray_frame_mean[t].at<float>(y, x), gray_frame_dev[t].at<float>(y, x), app_change[t].at<cv::Vec2f>(y, x)[0], app_change[t].at<cv::Vec2f>(y, x)[1]);

			float motion_similarity = cv::norm(v_flow - flow[t].at<cv::Vec2f>(y,x), norm_type)/motion_similarity_scale;
			//float dist_similarity = cv::norm(v_p - p, norm_type)/dist_similarity_scale; // TODO Normalization
			//float appearence_similarity = cv::norm(v_app - app, norm_type)/ appearence_similarity_scale;

			int index = x + y*video_resolution.width + (t-1)*video_resolution.area(); // skip first frame
			edges.emplace_back(v_index, index, motion_similarity);
			//edges.emplace_back(v_index, index, appearence_similarity);
			//edges.emplace_back(v_index, index, dist_similarity);
			//edges.emplace_back(v_index, index, cv::norm(cv::Vec3f(dist_similarity, motion_similarity, appearence_similarity), norm_type));
			//edges.emplace_back(v_index, index, cv::norm(cv::Vec3f(dist_similarity, motion_similarity), norm_type));
			//edges.emplace_back(v_index, index, cv::norm(cv::Vec3f(appearence_similarity, motion_similarity), norm_type));
		} } }
	} } }
	if(verbose) {
		std::cout << "Amount of vertices:\t" << vertices.size() << std::endl;
		std::cout << "Amount of edges:\t" << edges.size() << std::endl;
	}

	//// Segmentation: Kruskal`s minimum spanning tree
	// Create initial disjoint sets, each containg a singel vertex
	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map< int> parent_t;

	std::size_t vertices_count = vertices.size();;

	rank_t rank_map(vertices_count);
	parent_t parent_map(vertices_count);

	boost::disjoint_sets<rank_t, parent_t > dsets(rank_map, parent_map);
	for(auto vertex=vertices.begin(); vertex != vertices.end(); ++vertex) {
		dsets.make_set(*vertex);
	}

	// Create Minimum ST max weight and size property maps, the first initilized by zeros and the second by ones (i.e. initilal mst contain signle vertex)
	std::vector<double> mst_max_weight(vertices_count, 0);
	std::vector<double> mst_size(vertices_count, 1);

	// Segment the graph into disjoint sets, such that flow vectors form the same set have similar orientation
	edges.sort(); // Sort in ascending order

	for(auto edge = edges.begin(); edge != edges.end(); ++edge) {
		int parent_u = dsets.find_set(edge->_u);
		int parent_v = dsets.find_set(edge->_v);
		
		if( parent_u != parent_v &&
			edge->_weight < std::min(
						mst_max_weight[parent_u] + thi / mst_size[parent_u], // Similarity criteria
						mst_max_weight[parent_v] + thi / mst_size[parent_v])
					) {
			dsets.link(parent_u, parent_v); // Equivalent to union(u, v)

		 	int parent = dsets.find_set(parent_u);
			mst_max_weight[parent] = std::max(edge->_weight, std::max(mst_max_weight[parent_u], mst_max_weight[parent_v]));
			mst_size[parent] = mst_size[parent_u] + mst_size[parent_v];
		}	
	}
	if(verbose) {
		std::cout << "Amount of segments:\t" << dsets.count_sets(vertices.begin(), vertices.end()) << std::endl;
	}

	//// Return output
	// For the component colouring
	std::vector<int> random_numbers;
	unsigned int seed = rand() % 1000000;
	randomPermuteRange(pow(256,3), random_numbers, &seed); 

	// Do not print the first and the last frames
	std::vector<cv::Mat> output(frame.size()-2);
	for(size_t i=0; i<output.size(); ++i) {
		cv::Mat temp = cv::Mat::zeros(video_resolution, CV_8UC3);
		temp.copyTo(output[i]);
	}

	for(auto vertex = vertices.begin(); vertex != vertices.end(); ++vertex) {
		int t = *vertex / video_resolution.area();
		int y = (*vertex % video_resolution.area()) / video_resolution.width;
		int x = (*vertex % video_resolution.area()) % video_resolution.width;

		output[t].at<cv::Vec3b>(y,x) = colour_8UC3( random_numbers[dsets.find_set(*vertex)] );
	}

	for(size_t i=0; i < output.size()-1; ++i) {
		std::string output_filename = base_output_filename + '_' + std::to_string(i) + ".png";
		cv::imwrite(output_filename, output[i]);
	}

	return 0;
}

double similarity(const float beta, float scale_factor, float eps)
{
	return 1 - pow(EPS, beta/scale_factor) + eps;
}

double similarity(const int beta, float scale_factor, float eps)
{
	return 1 - pow(EPS, beta/scale_factor) + eps;
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
