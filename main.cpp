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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

//float scalar(float rx, float ry, float lx, float ly);
float scalar(const unsigned char rx, const unsigned char ry, const unsigned char lx, const unsigned char ly);
//float norm(float x, float y);
float norm(const unsigned char x, const unsigned char y);

int linear_index(int i, int j, int width);
cv::Point matrix_index(int k, int width);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

// TODO Let flow range be [0 2^16]
int main(int argc, char * argv[])
{
	const float PI = 3.1415926;
	const float alpha = 0.197;
	const float gamma = 50.0;
	const float scale_factor = 0.6;
	const int inner_iteratoins = 10;
	const int outter_iterations = 77;
	const int solver_iterations = 10;
 
	if( argc != 5 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1 - frame 1" << std::endl;
		std::cout << "2 - frame 2" << std::endl;
		std::cout << "3 - output filename" << std::endl;
		std::cout << "4 - eps max magnitude of a flow vector to be ignored [0 - min magnitude, 255 - max magnitude]" << std::endl;
		std::cout << "5 - thi max angle in degrees of two flow vectors to be considered as co-directed" << std::endl;
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

	float eps = atof(argv[4]);
	float thi_rad = (PI*atoi(argv[5]))/180.0;

	if( thi_rad < 0 || thi_rad > PI/2 ) {
		std::cout << "The thi should be between 0 and 90 degrees" << std::endl;
		return 0;
	}

	// Convert the input to CV_32FC1, as required for BroxOpticalFlow
	for(int i=0; i<2; ++i) {
		cv::cvtColor(frame[i], frame[i], cv::COLOR_BGR2GRAY);
		frame[i].convertTo(frame[i], CV_32F, 1.0 /255.0);
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
		flow[i].convertTo(flow[i], CV_8U, 255.0);
	}

	//// Create a graph of flows vectors with edges corresponding to the given affinity
	std::list<std::pair<int, int> > edges;
	cv::Size size = flow[0].size();
	std::vector<bool> non_null_flow_map(size.height*size.width, false);

	//std::cout << norm(*flow[0].ptr<double>(size.height), *flow[1].ptr<double>(size.height)) << std::endl;

	int counter = 0;
	// Left and bottom optical flow borders are ignored in sake of simplicity
	for(int j=0; j<size.height-1; ++j) {
		const unsigned char * it0 = flow[0].ptr<const unsigned char>(j), * it1 = flow[1].ptr<const unsigned char>(j);
		const unsigned char * bot_it0 = flow[0].ptr<const unsigned char>(j+1), * bot_it1 = flow[1].ptr<const unsigned char>(j+1);

		for(int i=0; i<size.width-1; ++i, ++it0, ++it1, ++bot_it0, ++bot_it1) {
			if(norm(*it0, *it1) < eps) {
				// Skip null-flow vectors
				continue;
			}
			counter++;

			non_null_flow_map[linear_index(i, j, size.width)] = true;
	
			float scalar_product = scalar(*it0, *it1, *(it0+1), *(it1+1));
			// If vector flows is co-directed
			if( scalar_product > 0 && (scalar_product / (norm(*it0, *it1) * norm(*(it0+1),*(it1+1)))) >= cos(thi_rad) ) {
				edges.emplace_back(std::make_pair(linear_index(i, j, size.width), linear_index(i+1, j, size.width)));
			}

			scalar_product = scalar(*it0, *it1, *bot_it0, *bot_it1);
			// If vector flows is co-directed
			if( scalar_product > 0 && (scalar_product / (norm(*it0, *it1) * norm(*bot_it0,*bot_it1))) >= cos(thi_rad) ) {
				edges.emplace_back(std::make_pair(linear_index(i, j, size.width), linear_index(i, j+1, size.width)));
			}
		}
	}
	std::cout << "Amount of not filterd flow vectors: " << counter << std::endl;
	std::cout << "Amount of connections: " << edges.size() << std::endl;

	//// Extract connected components
	boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph(edges.begin(), edges.end(), size.width*size.height);

	std::vector<int> component_map(boost::num_vertices(graph));
	int connected_components_count = boost::connected_components(graph, &component_map[0]);
	std::cout << "Connected components: " << connected_components_count << std::endl;

	//// Return optical flow
	cv::Mat output_flow(size, CV_16UC1, cv::Scalar_<unsigned short>(0));

	for(int j=0; j<size.height; ++j) {
		const unsigned char * src0 = flow[0].ptr<const unsigned char>(j),
	       			* src1 = flow[1].ptr<const unsigned char>(j);
		unsigned short * dst = output_flow.ptr<unsigned short>(j);

		for(int i=0; i<size.width; ++i) {
			dst[i] = norm(src0[i], src1[i]);
		}
	}

	cv::imwrite(std::string("output_flow.jpg"), output_flow);

	//// Return output
	// For the component colouring
	std::vector<int> random_numbers;
	unsigned int seed = rand() % 1000000;
	randomPermuteRange(pow(256,3), random_numbers, &seed); 

	cv::Mat output(size, CV_8UC3, cv::Scalar_<unsigned char>(0));

	size_t k=0;
	for(cv::MatIterator_<cv::Vec3b> dst = output.begin<cv::Vec3b>();
       		dst!=output.end<cv::Vec3b>();
	       	++dst) {
		if(!non_null_flow_map[k]) {
			++k;
			continue;
		}

		*dst = colour_8UC3( random_numbers[component_map[k++]] );
	}
	std::cout << k << std::endl;

	cv::imwrite(argv[3], output);
	
	return 0;
}

float scalar(float rx, float ry, float lx, float ly)
{
	return rx*lx + ry*ly;
}

float scalar(const unsigned char rx, const unsigned char ry, const unsigned char lx, const unsigned char ly)
{
	return rx*lx + ry*ly;
}

float norm(const unsigned char x, const unsigned char y)
{
	return sqrt(pow((const unsigned short)x,2) + pow((const unsigned short)y,2));
}

float norm(float x, float y)
{
	return sqrt(pow(x,2) + pow(y,2));
}

int linear_index(int i, int j, int width)
{
	return i + j*width;
}

cv::Point matrix_index(int k, int width)
{
	//return cv::Point( (int)k/width, k%width );
	return cv::Point( k%width, (int)k/width );
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
