#include <iostream>
#include <cmath> // abs pow sqrt
#include <list>
#include <utility> // pair
#include <vector>
#include <stdlib.h> // rand_r
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/property_map/vector_property_map.hpp>
#include <boost/pending/disjoint_sets.hpp>

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
	const float alpha = 0.197;
	const float gamma = 50.0;
	const float scale_factor = 0.6;
	const int inner_iteratoins = 10;
	const int outter_iterations = 77;
	const int solver_iterations = 10;
 
	// TODO Load optical flow file (.flo) instead of computing one
	if( argc != 5 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1 - frame 1" << std::endl;
		std::cout << "2 - frame 2" << std::endl;
		std::cout << "3 - output filename" << std::endl;
		std::cout << "4 - thi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "5 - eps min allowable norm of a flow vector; only vectors with at least eps norm are considered" << std::endl;
		std::cout << "Optical flow parameters are hardcoded" << std::endl;
		return 1;
	}

	//// Read and check input
	cv::Mat frame[2];
	frame[0] = cv::imread(argv[1]);
	frame[1] = cv::imread(argv[2]);
	if(frame[0].size() != frame[1].size()) {
		std::cout << "The input frames have not equal sizes!" << std::endl;
		return 0;
	}
	if(frame[0].type() != frame[1].type()) {
		std::cout << "The input frames are of different types!" << std::endl;
		return 0;
	}
	// Convert the input to CV_32FC1, as required for BroxOpticalFlow
	for(int i=0; i<2; ++i) {
		cv::cvtColor(frame[i], frame[i], cv::COLOR_BGR2GRAY);
		frame[i].convertTo(frame[i], CV_32F, 1.0 /255.0);
	}

	std::string output_filename(argv[3]);

	double thi = atof(argv[4]);
	if( thi < 0 ) {
		std::cout << "The thi should be non-negative" << std::endl;
		return 0;
	}

	double eps = atof(argv[5]);
	if( eps < 0 || 1 < eps ) {
		std::cout << "The eps should be in [0;1] range" << std::endl;
		return 0;
	}

	//// Compute an optical flow
	cv::gpu::GpuMat gpuFrame[2], gpuFlow[2];
	for(int i=0; i<2; ++i) {
		gpuFrame[i].upload(frame[i]);
	}

	// The output is of the same type as the input
	cv::gpu::BroxOpticalFlow bof(alpha, gamma, scale_factor, inner_iteratoins, outter_iterations, solver_iterations);
	bof(gpuFrame[0], gpuFrame[1], gpuFlow[0], gpuFlow[1]);

	cv::Mat flow[2];
	for(int i=0; i<2; ++i) {
		gpuFlow[i].download(flow[i]);
	}

	//// Create a graph of flow vectors with edges connecting neightbour vertices,
	//// scaled by a degree of angle between them  
	cv::Size flow_size = flow[0].size();

	std::size_t vertices_count = flow_size.width * flow_size.height;

	std::list< int > vertices;
	for(int j=0; j<flow_size.height; ++j) {
		const float * x_flow = flow[0].ptr<float>(j);
		const float * y_flow = flow[1].ptr<float>(j);

		for(int i=0; i<flow_size.width; ++i, ++x_flow, ++y_flow) {
			//std::cout << i << ' ' << j << std::endl;
			//std::cout << "|" << std::endl;
			if(norm(*x_flow, *y_flow) > eps) {
				vertices.push_back(i + j*flow_size.width);
			}
		}
	}

	std::list< edge_t > edges;
	// Add edges along x axis
	for(int j=0; j<flow_size.height; ++j) {
		const float * x_flow = flow[0].ptr<float>(j);
		const float * y_flow = flow[1].ptr<float>(j);

		for(int i=0; i<flow_size.width-1; ++i, ++x_flow, ++y_flow) {
			double s = scalar(*x_flow, *y_flow, *(x_flow+1), *(y_flow+1));
			double norm_a = norm(*x_flow, *y_flow);
			double norm_b = norm(*(x_flow+1), *(y_flow+1));

			if(norm_a > eps && norm_b > eps) {
				edges.emplace_back(i + j*flow_size.width, i+1 + j*flow_size.width, acos(s / (norm_a * norm_b)));
			}
		}
	}

	// Add edges along x axis
	for(int j=0; j<flow_size.height-1; ++j) {
		const float * x_flow = flow[0].ptr<float>(j);
		const float * y_flow = flow[1].ptr<float>(j);
		const float * x_flow_bot = flow[0].ptr<float>(j+1);
		const float * y_flow_bot = flow[1].ptr<float>(j+1);

		for(int i=0; i<flow_size.width; ++i, ++x_flow, ++y_flow, ++x_flow_bot, ++y_flow_bot) {
			double s = scalar(*x_flow, *y_flow, *x_flow_bot, *y_flow_bot);
			double norm_a = norm(*x_flow, *y_flow);
			double norm_b = norm(*x_flow_bot, *y_flow_bot);

			if(norm_a > eps && norm_b > eps) {
				edges.emplace_back(i + j*flow_size.width, i + (j+1)*flow_size.width, acos(s / (norm_a * norm_b)));
			}
		}
	}
	//std::cout << edges.size() << std::endl;


	//// Create initial disjoint sets, each containg a singeltone vertex
	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map< int> parent_t;

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
	std::cout << "Amount of segments: " << dsets.count_sets(vertices.begin(), vertices.end()) << std::endl;

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
