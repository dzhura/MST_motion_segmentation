#include <iostream>
#include <cmath> // abs pow sqrt
#include <list>
#include <utility> // pair
#include <vector>
#include <stdio.h> // C-style file reading
#include <stdlib.h> // rand_r
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp> // split

#include <boost/property_map/vector_property_map.hpp>
#include <boost/pending/disjoint_sets.hpp>

#define EPS 2.7182818

const bool verbose=1;
const bool show_optical_flow=0;

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

double scalar(const double rx, const double ry, const double lx, const double ly);

double similarity(const float beta, float scale_factor, float eps=0);
double similarity(const int beta, float scale_factor, float eps=0);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
 
	// TODO Load optical flow file (.flo) instead of computing one
	if( argc != 10 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1 - image file" << std::endl;
		std::cout << "2 - .flo file" << std::endl;
		std::cout << "3 - (0,inf)\t\tscaling factor for distance similarity" << std::endl;
		std::cout << "4 - (0,inf)\t\tcaling factor for appearence similarity" << std::endl;
		std::cout << "5 - (0,inf)\t\tscaling factor for motion similarity" << std::endl;
		std::cout << "6 - [0,1)\t\teps for motion similarity" << std::endl;
		std::cout << "7 - 2*n-1, n=1,2,3...\tregion size" << std::endl;
		std::cout << "8 - [0,inf)\t\tthi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "9 - 2*n+1, n=1,2,3...\tsize of window, where all elements are linked with the central" << std::endl;
		std::cout << "10 - output file" << std::endl;
		return 1;
	}

	//// Input read, check and pre processings
	cv::Mat image = cv::imread(argv[1]);
	if(!image.data) {
		std::cout << "Failed to read image " << argv[1] << std::endl;
		return 1;
	}
	if(image.depth() != CV_8U && image.depth() != CV_16U) {
		std::cout << "No appropriate image depth" << std::endl;
		std::cout << "It should be unsigned 8-bit or 16-bit depth" << std::endl;
		return 1;
	}

	cv::Mat flow;
	if( !read_opticalFlow(std::string(argv[2]), flow)) {
		std::cout << "Could not read opticalflow: " << argv[2] << std::endl;
		return 1;
	}

	float dist_similarity_scale = atof(argv[3]);
	if(dist_similarity_scale <= 0) {
		std::cout << "Scaling factor for distance similarity should be positive" << std::endl;
		return 1;
	}

	float appearence_similarity_scale = atof(argv[4]);
	if(appearence_similarity_scale <= 0) {
		std::cout << "Scaling factor for appearence similarity should be positive" << std::endl;
		return 1;
	}

	float motion_similarity_scale = atof(argv[5]);
	if(motion_similarity_scale <= 0) {
		std::cout << "Scaling factor for motion similarity should be positive" << std::endl;
		return 1;
	}

	float motion_eps = atof(argv[6]);
	if(!( 0 <= motion_eps && motion_eps < 1)) {
		std::cout << "Eps for motion similarity should be in [0, 1) range" << std::endl;
		return 1;
	}

	int neighbourhood_size = atoi(argv[7]);
	if(!(0 < neighbourhood_size && neighbourhood_size%2 == 1)) {
		std::cout << "Neighbourhood size should be positive and odd" << std::endl;
		return 1;
	}

	double thi = atof(argv[8]);
	if( thi < 0 ) {
		std::cout << "The thi should be non-negative" << std::endl;
		return 0;
	}

	int window_size = atoi(argv[9]);
	if( !(3 <= window_size && window_size%2 == 1) ) {
		std::cout << "The size of window should be larger 3 and be odd " << window_size << std::endl;
		return 0;
	}

	std::string output_filename(argv[10]);

	//// Preprocessing
	// Normalize image
	switch(image.depth()) {
		case CV_8U:
			image.convertTo(image, CV_32F, 1.0/255.0);
			break;
		case CV_16U:
			image.convertTo(image, CV_32F, 1.0/65535.0);
			break;
	}
	// Convert image to gray scale
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, CV_RGB2GRAY);

	// Convert optical flow to polar coordiante
	float max_flow_magnitude = 0;
	float max_flow_angle = 0;
	for(auto flow_elm = flow.begin<cv::Vec2f>(); flow_elm != flow.end<cv::Vec2f>(); ++flow_elm) {
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
	// Normalize optical flow
	for(auto flow_elm = flow.begin<cv::Vec2f>(); flow_elm != flow.end<cv::Vec2f>(); ++flow_elm) {
		if( (*flow_elm)[0] > 1e9 || (*flow_elm)[1] > 1e9 ) { // if flow value is unknow
			continue;
		}

		(*flow_elm)[0] /= max_flow_magnitude; // in [0;1]
		(*flow_elm)[1] /= max_flow_angle; // in (0;1]
	}

	// Compute mean and st deviation of the image
	// TODO test it
	cv::Mat kernel = cv::Mat::ones(neighbourhood_size, neighbourhood_size, CV_32F) / (float)pow(neighbourhood_size, 2);

	cv::Mat gray_image_mean(gray_image.size(), CV_32FC1);
	cv::filter2D(gray_image, gray_image_mean, CV_32F, kernel);

	cv::Mat gray_image_dev(gray_image.size(), CV_32FC1);
	cv::pow(gray_image - gray_image_mean, 2, gray_image_dev);
	cv::filter2D(gray_image_dev, gray_image_dev, CV_32F, kernel);
	cv::pow(gray_image_dev, 0.5, gray_image_dev);

	//// Create a graph model
	std::list< int > vertices;
	std::list< edge_t > edges;

	cv::Size flow_size = flow.size();
	for(int v_y=0; v_y < flow_size.height; ++v_y) {

		for(int v_x=0; v_x < flow_size.width; ++v_x) {

			vertices.push_back(v_x + v_y*flow_size.width);

			cv::Vec2f v_flow = *(v_x + flow.ptr<cv::Vec2f>(v_y));
			//cv::Vec2i v_p(v_x, v_y);
			//cv::Vec2f v_app(*(v_x + gray_image_mean.ptr<float>(v_y)), *(v_x + gray_image_dev.ptr<float>(v_y)));

			for(int y = std::max(0, v_y - window_size/2); y < std::min(flow_size.height, v_y+1 + window_size/2); ++y) {

				const cv::Vec2f * p_flow = flow.ptr<cv::Vec2f>(y);
				//const float * p_mean = gray_image_mean.ptr<float>(y);
				//const float * p_dev = gray_image_dev.ptr<float>(y);

				for(int x = std::max(0, v_x - window_size/2); x < std::min(flow_size.width, v_x+1 + window_size/2); ++x) {
				
					//cv::Vec2i p(x, y);
					//cv::Vec2f app(*(p_mean + x), *(p_dev + x));

					float motion_similarity = cv::norm(v_flow - *(p_flow + x), norm_type)/motion_similarity_scale;
					//float dist_similarity = cv::norm(v_p - p, norm_type)/(window_size*dist_similarity_scale*sqrt(2)/2); // Normalized; in [0;1] range
					//float appearence_similarity = cv::norm(v_app - app, norm_type)/ appearence_similarity_scale;

					edges.emplace_back(v_x + v_y*flow_size.width, x + y*flow_size.width, motion_similarity);
					//edges.emplace_back(v_x + v_y*flow_size.width, x + y*flow_size.width, cv::norm(cv::Vec3f(dist_similarity, motion_similarity), norm_type));
					//edges.emplace_back(v_x + v_y*flow_size.width, x + y*flow_size.width, cv::norm(cv::Vec3f(appearence_similarity, motion_similarity), norm_type));
					//edges.emplace_back(v_x + v_y*flow_size.width, x + y*flow_size.width, cv::norm(cv::Vec3f(dist_similarity, motion_similarity, appearence_similarity), norm_type));
				}
			}
		}
	}

	if(verbose) {
		std::cout << "Amount of vertices:\t" << vertices.size() << std::endl;
		std::cout << "Amount of edges:\t" << edges.size() << std::endl;
	}

	//// Segmentation
	// Create initial disjoint sets, each containg a singel vertex
	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map< int> parent_t;

	std::size_t vertices_count = flow_size.area();
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
						mst_max_weight[parent_u] + thi / mst_size[parent_u],
						mst_max_weight[parent_v] + thi / mst_size[parent_v])
					) {
			dsets.link(parent_u, parent_v); // Equivalent to union(u, v)

		 	int parent = dsets.find_set(parent_u);
			mst_max_weight[parent] = std::max(edge->_weight, std::max(mst_max_weight[parent_u], mst_max_weight[parent_v])); // due to the condition above
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

	cv::Mat output(flow_size, CV_8UC3, cv::Scalar_<unsigned char>(0));

	for(auto vertex = vertices.begin(); vertex != vertices.end(); ++vertex) {
		int i = *vertex / flow_size.width;
		int j = *vertex % flow_size.width;

		output.at<cv::Vec3b>(i,j) = colour_8UC3( random_numbers[dsets.find_set(*vertex)] );
	}

	cv::imwrite(output_filename, output);

	if(show_optical_flow) {
		// Flow for printing
		cv::Mat flow_output[2];

		cv::split(flow, flow_output);

		cv::imshow("optical flow u", flow_output[0]);	
		cv::imshow("optical flow v", flow_output[1]);	
		cv::waitKey(0);
	}
	
	return 0;
}

double scalar(const double rx, const double ry, const double lx, const double ly)
{
	return rx*lx + ry*ly;
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
