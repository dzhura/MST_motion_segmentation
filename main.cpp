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

const bool verbose=1;
const bool show_optical_flow=0;

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
double norm(const double x, const double y);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

int main(int argc, char * argv[])
{
	//const float PI = 3.1415926;
 
	// TODO Load optical flow file (.flo) instead of computing one
	if( argc != 4 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1 - .flo file" << std::endl;
		std::cout << "2 - output filename" << std::endl;
		std::cout << "3 - thi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "4 - eps min allowable norm of a flow vector; only vectors with at least eps norm are considered" << std::endl;
		return 1;
	}

	//// Read and check optical flow
	std::string input_flo_filename(argv[1]);

	if(input_flo_filename.empty()) {
		std::cout << "Input flo file is empty!" << std::endl;
		return 0;
	}

	FILE * stream = fopen(input_flo_filename.c_str(), "rb");
	if(stream == 0) {
		std::cout << "Could not open " << input_flo_filename << std::endl;
		return 0;
	}

	int width, height;
	float tag;
	if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
	       	(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
	       	(int)fread(&height, sizeof(int),   1, stream) != 1) {
		std::cout << "Problem reading file " << input_flo_filename << std::endl;
		return 0;
	}

	if (tag != 202021.25) { // simple test for correct endian-ness
		std::cout << "Wrong tag (possibly due to big-endian machine?)" << std::endl;
		return 0;
	}

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999) {
		std::cout << "Illegal width " << input_flo_filename << std::endl;
		return 0;
	}

	if (height < 1 || height > 99999) {
		std::cout << "Illegal height " << input_flo_filename << std::endl;
		return 0;
	}

	cv::Size flow_size(width, height);

	size_t flo_data_size = 2 * flow_size.area() * sizeof(float);
	unsigned char * flo_data = new unsigned char[flo_data_size];

	size_t ret_code = fread(flo_data, sizeof(float), 2*flow_size.area(), stream);

	if(ret_code != 2*(size_t)flow_size.area()) {
		if(feof(stream)) {
			std::cout << "Error reading " << input_flo_filename << ": unexpected end of the file" << std::endl;
			return 0;
		}
		else if(ferror(stream)) {
			std::cout << "Error reading " << input_flo_filename << std::endl;
			return 0;
		}
	}

	if (fgetc(stream) != EOF) {
		std::cout << "File is too long" << input_flo_filename << std::endl;
		return 0;
	}

	fclose(stream);

	cv::Mat flow(flow_size, CV_32FC2, flo_data);



	//// Read and check the rest of parameters
	double thi = atof(argv[3]);
	if( thi < 0 ) {
		std::cout << "The thi should be non-negative" << std::endl;
		return 0;
	}

	double eps = atof(argv[4]);
	if( eps < 0 || 1 < eps ) {
		std::cout << "The eps should be in [0;1] range" << std::endl;
		return 0;
	}

	std::string output_filename(argv[2]);

	//// Create a graph of flow vectors with edges connecting neightbour vertices,
	//// scaled by a degree of angle between them  
	std::list< int > vertices;
	for(int j=0; j<flow_size.height; ++j) {
		const float * p_flow = flow.ptr<float>(j);

		// NB: p_flow = x-flow component, p_flow+1 = y-flow component
		for(int i=0; i<flow_size.width; ++i, p_flow+=2) {
			//std::cout << i << ' ' << j << std::endl;
			//std::cout << "|" << std::endl;
			if(norm(*p_flow, *(p_flow+1)) > eps) {
				vertices.push_back(i + j*flow_size.width);
			}
		}
	}

	std::list< edge_t > edges;
	
	// Add edges along x axis
	for(int j=0; j<flow_size.height; ++j) {
		const float * p_flow = flow.ptr<float>(j);

		for(int i=0; i<flow_size.width-1; ++i, p_flow+=2) {
			double s = scalar(*p_flow, *(p_flow+1), *(p_flow+2), *(p_flow+3));
			double norm_a = norm(*p_flow, *(p_flow+1));
			double norm_b = norm(*(p_flow+2), *(p_flow+3));

			if(norm_a > eps && norm_b > eps) {
				edges.emplace_back(i + j*flow_size.width, i+1 + j*flow_size.width, acos(s / (norm_a * norm_b)));
			}
		}
	}

	// Add edges along x axis
	for(int j=0; j<flow_size.height-1; ++j) {
		const float * p_flow = flow.ptr<float>(j);
		const float * p_flow_bot = flow.ptr<float>(j+1);

		for(int i=0; i<flow_size.width; ++i, p_flow+=2, p_flow_bot+=2) {
			double s = scalar(*p_flow, *(p_flow+1), *p_flow_bot, *(p_flow_bot+1));
			double norm_a = norm(*p_flow, *(p_flow+1));
			double norm_b = norm(*p_flow_bot, *(p_flow_bot+1));

			if(norm_a > eps && norm_b > eps) {
				edges.emplace_back(i + j*flow_size.width, i + (j+1)*flow_size.width, acos(s / (norm_a * norm_b)));
			}
		}
	}
	//std::cout << edges.size() << std::endl;


	//// Create initial disjoint sets, each containg a singeltone vertex
	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map< int> parent_t;

	std::size_t vertices_count = flow_size.area();
	rank_t rank_map(vertices_count);
	parent_t parent_map(vertices_count);

	boost::disjoint_sets<rank_t, parent_t > dsets(rank_map, parent_map);
	for(auto vertex=vertices.begin(); vertex != vertices.end(); ++vertex) {
		dsets.make_set(*vertex);
	}

	//// Create MST max weight and size property maps, the first initilized by zeros and the second by ones (i.e. initilal mst contain signle vertex)
	std::vector<double> mst_max_weight(vertices_count, 0);
	std::vector<double> mst_size(vertices_count, 1);

	//// Segment the graph into disjoint sets, such that flow vectors form the same set have similar orientation
	edges.sort(); // Sort in ascending order

	for(auto edge = edges.begin(); edge != edges.end(); ++edge) {
		 int parent_u = dsets.find_set(edge->_u);
		 int parent_v = dsets.find_set(edge->_v);
		
		if( parent_u != parent_v &&
			edge->_weight < std::min(
						mst_max_weight[parent_u] + thi / mst_size[parent_u],
						mst_max_weight[parent_v] + thi / mst_size[parent_v])) {
			dsets.link(parent_u, parent_v); // Equivalent to union(u, v)

		 	int parent = dsets.find_set(parent_u);
			mst_max_weight[parent] = std::max(mst_max_weight[parent_u], mst_max_weight[parent_v]);
			mst_size[parent] = mst_size[parent_u] + mst_size[parent_v];
		}	
	}
	if(verbose) {
		std::cout << "Amount of segments: " << dsets.count_sets(vertices.begin(), vertices.end()) << std::endl;
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
	
	delete[] flo_data;
 
	return 0;
}

double scalar(const double rx, const double ry, const double lx, const double ly)
{
	return rx*lx + ry*ly;
}

double norm(const double x, const double y)
{
	return sqrt(pow(x,2) + pow(y,2));
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
