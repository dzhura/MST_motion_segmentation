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

float dist_similarity_scale = 1;
float motion_similarity_scale = 1;
int window_size = 1;

double scalar(const double rx, const double ry, const double lx, const double ly);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
 
	// TODO Load optical flow file (.flo) instead of computing one
	if( argc != 2 + 1) {
		std::cerr << argv[0] <<" usage: <.flo file> <thi>" << std::endl;
		std::cerr << "result is returned to standard output" << std::endl;
		return 2;
	}

	//// Input read, check and pre processings
	cv::Mat flow;
	if( !read_opticalFlow(std::string(argv[1]), flow)) {
		std::cerr << "Could not read opticalflow: " << argv[2] << std::endl;
		return 1;
	}

	double thi = atof(argv[2]);


	//// Preprocessing

	// Normalize optical flow

	//// Create initial graph G(V, E), |V| == total count of flow vectors
	std::list< int > vertices;
	std::list< edge_t > edges;

	cv::Size flow_size = flow.size();
	for(int v_y=0; v_y < flow_size.height; ++v_y) {

		for(int v_x=0; v_x < flow_size.width; ++v_x) {

			vertices.push_back(v_x + v_y*flow_size.width);

			cv::Vec2f v_flow = flow.at<cv::Vec2f>(v_y, v_x);
			for(int dy = 1; dy < std::max(2, 1+window_size/2) && (v_y + dy < flow_size.height); ++dy) {
				int dx = 0;
				float motion_similarity = cv::norm(v_flow - flow.at<cv::Vec2f>(v_y+dy, v_x+dx), norm_type)/motion_similarity_scale;
				edges.emplace_back(v_x + v_y*flow_size.width, v_x+dx + (v_y+dy)*flow_size.width, motion_similarity);
			}
			for(int dx = 1; dx < std::max(2, 1+window_size/2) && (v_x + dx < flow_size.width); ++dx) {
				int dy = 0;
				float motion_similarity = cv::norm(v_flow - flow.at<cv::Vec2f>(v_y+dy, v_x+dx), norm_type)/motion_similarity_scale;
				edges.emplace_back(v_x + v_y*flow_size.width, v_x+dx + (v_y+dy)*flow_size.width, motion_similarity);
			}
			for(int dy = 1; dy < std::max(2, 1+window_size/2) && (v_y + dy < flow_size.height); ++dy) {
				for(int dx = 1; dx < std::max(2, 1+window_size/2) && (v_x + dx < flow_size.width); ++dx) {
					float motion_similarity = cv::norm(v_flow - flow.at<cv::Vec2f>(v_y+dy, v_x+dx), norm_type)/motion_similarity_scale;
					edges.emplace_back(v_x + v_y*flow_size.width, v_x+dx + (v_y+dy)*flow_size.width, motion_similarity);
				}
			}
		}
	}
	
	//// Segmentation
	// Create initial disjoint sets, each containg a singel vertex
	typedef boost::vector_property_map<std::size_t> rank_t;
	typedef boost::vector_property_map< int> parent_t;

	std::size_t vertices_count = vertices.size();

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

	//// For each cluster, compute sum and count of flow vectors
	std::vector<cv::Vec2f> sum_of_flows(vertices_count, cv::Vec2f(0,0));
	std::vector<size_t> cluster_sizes(vertices_count, 0);
	for(auto vertex = vertices.begin(); vertex != vertices.end(); ++vertex) {
		int y = *vertex / flow_size.width;
		int x = *vertex % flow_size.width;
		size_t cluster_id = dsets.find_set(*vertex);

		sum_of_flows[cluster_id] += flow.at<cv::Vec2f>(y,x);
		cluster_sizes[cluster_id]++;
	}

	//// Return output
	size_t cluster_count = 0;
	for( size_t i=0; i<vertices_count; ++i) {
		if( cluster_sizes[i] != 0 ) {
			cluster_count++;
		}
	}

	std::cout << cluster_count << std::endl;
	for( size_t i=0; i<vertices_count; ++i) {
		if( cluster_sizes[i] == 0 ) {
			continue;
		}
		cv::Vec2f avrg_flow = sum_of_flows[i] * (1.0 / cluster_sizes[i]);
		std::cout << avrg_flow[0] << ' ' << avrg_flow[1] << ' ';
	}
	std::cout << std::endl;

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

	std::string output_filename = "out_flow-thi" + std::to_string(thi) + ".png";
	cv::imwrite(output_filename.c_str(), output);

	return 0;
}

double scalar(const double rx, const double ry, const double lx, const double ly)
{
	return rx*lx + ry*ly;
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
