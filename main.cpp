#include <iostream>
#include <cmath> // abs pow sqrt M_PI
#include <list>
#include <vector>
#include <map>
#include <set> // for searching unique elements
#include <unordered_set> // for computing Jaccard similarity
#include <algorithm> // for_each
#include <utility> // pair
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


// TODO fill image
struct binary_image
{
	cv::Size _size;
	std::vector<boost::dynamic_bitset<>> _bimage;

	public:

	binary_image()
	{ }

	binary_image(const cv::Size & size):
		_size(size), _bimage(_size.width, boost::dynamic_bitset<>(_size.height))
	{ }

	binary_image(const std::list<int> & segment_points, const cv::Size & size):
		binary_image(size)
	{
		_fill_by(segment_points);
	}

	void create(const std::list<int> & segment_points, const cv::Size & size)
	{
		_size = size;

		_bimage.resize(_size.width, boost::dynamic_bitset<>(_size.height));
		_reset();
		_fill_by(segment_points);
	}

	binary_image operator& (const binary_image & ro) const
	{
		assert(_size == ro._size);

		binary_image result(_size);
		for(size_t i=0; i < _bimage.size(); ++i) {
			result._bimage[i] = _bimage[i] & ro._bimage[i];
		}
		return result;
	}
	binary_image operator| (const binary_image & ro) const
	{
		assert(_size == ro._size);

		binary_image result(_size);
		for(size_t i=0; i < _bimage.size(); ++i) {
			result._bimage[i] = _bimage[i] | ro._bimage[i];
		}
		return result;
	}

	size_t count() const
	{
		size_t result=0;
		for(size_t i=0; i < _bimage.size(); ++i) {
			result += _bimage[i].count();
		}
		return result;
	}

	binary_image move(int x, int y) const 
	{
		binary_image result(*this);

		int dir_x = (x < 0)? -1: 1;
		int dir_y = (y < 0)? -1: 1;

		result._move_along_x(std::abs(x), dir_x);
		result._move_along_y(std::abs(y), dir_y);

		return result;
	}

	private:
	void _move_along_x(int x, int dir) {
		if(dir > 1) { 	// move right
			for(int i=_size.width-1; i>=x; --i) {
				_bimage[i] = _bimage[i-x];
			}
			for(int i=0; i<x; ++i) {
				_bimage[i] = boost::dynamic_bitset<>(_size.height); // fill by empty cols
			}
		}
		else {		// move left
			for(int i=0; i<_size.width - x; ++i) {
				_bimage[i] = _bimage[i+x];
			}
			for(int i=_size.width - x; i<_size.width; ++i) {
				_bimage[i] = boost::dynamic_bitset<>(_size.height); // fill by empty cols
			}
		}
	}
	// Works ONLY for little endian
	void _move_along_y(int y, int dir)
       	{
		if(dir > 1) {	// move down
			for(int i=0; i<_size.width; ++i) {
				_bimage[i] >>=y;
			}
		}
		else {		// move up
			for(int i=0; i<_size.width; ++i) {
				_bimage[i] <<=y;
			}
		}

	}

	void _reset()
	{
		for(int i=0; i<_size.width; ++i) {
			_bimage[i].reset();
		}
	}
	void _fill_by(const std::list<int> & pixels)
       	{
		for(std::list<int>::const_iterator p=pixels.begin(); p!=pixels.end(); ++p) {
			int x = (*p) % _size.width;
			int y = (*p) / _size.width;
			_bimage[x][y] = 1;
		}
	}
};

struct optical_flow_segment
{
	std::list<int> _points;
	binary_image _mask;

	int _avrg_dx;
	int _avrg_dy;

	optical_flow_segment()
       	{ }
};

struct trajectory
{
	int _start_frame;
	std::list<optical_flow_segment> _segments;

	trajectory(const optical_flow_segment & seg, int start_frame):
		_start_frame(start_frame)
	{
		assert(0 <= start_frame);
		add_segment(seg);
	}

	void add_segment(const optical_flow_segment & seg)
	{
		_segments.push_back(seg);
	}

	const optical_flow_segment & back_segment() const
	{
		return _segments.back();
	}
};

void average_displacement(const std::list<int> & points, const cv::Mat & flow, int & avrg_dx, int & avrg_dy);

// Returns nan is lo or ro are zero vectros
// TODO Use homogeneous coordinates to prevent divizion by zero
float cosine_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro);

float l2_similarity(const cv::Vec2f & lo, const cv::Vec2f & ro);

float Jaccard_similarity(const std::list<int> & lo, const std::list<int> & ro);

void randomPermuteRange(int n, std::vector<int>& vec, unsigned int *seed);
cv::Vec3b colour_8UC3(int index);

bool read_opticalFlow(const std::string & opticalflow_filename, cv::Mat & out);

int main(int argc, char * argv[])
{
	if( argc != 5 + 1) {
		std::cout << argv[0] <<" usage:" << std::endl;
		std::cout << "1\tfile list, including amount of frames, amount of frames for segmentation, frames to skip, frame paths and optical flow paths" << std::endl;
		std::cout << "2\t(0,inf)\t\tthi additional threshold; larger thi results in larger segments" << std::endl;
		std::cout << "3\t(0,1)\t\tthreshold for linking" << std::endl;
		std::cout << "4\t2*n+1, n=1,2,3...\tsize of spatial window, where all elements are linked with the central" << std::endl;
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

	int amount_of_frames=0, amount_of_frames_for_seg = 0, frames_to_skip=0;
	infile >> amount_of_frames >> amount_of_frames_for_seg >> frames_to_skip;

	assert(0 < amount_of_frames);
	assert(-1 < frames_to_skip);
	assert(0 < amount_of_frames_for_seg);
	assert(amount_of_frames_for_seg + frames_to_skip <= amount_of_frames);

	// Skip first frames
	for(int i=0; i<frames_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	std::vector<cv::Mat> frame(amount_of_frames_for_seg);
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
	for(size_t i=0; i < amount_of_frames - frames_to_skip - frame.size(); ++i) {
		std::string dummy;
		infile >> dummy;
	}

	// Skip first opt flows
	for(int i=0; i<frames_to_skip; ++i) {
		std::string dummy;
		infile >> dummy;
	}

	std::vector<cv::Mat> flow(amount_of_frames_for_seg);
	for(int i=0; i<amount_of_frames_for_seg; ++i) {
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

	double thi = atof(argv[2]);
	assert( 0 < thi );

	double link_threshold = atof(argv[3]);
	assert(0 < link_threshold && link_threshold < 1);

	int spatial_window_size = atoi(argv[4]);
	assert( 3 <= spatial_window_size && spatial_window_size < cv::min(frame[0].cols, frame[0].rows) && spatial_window_size%2 == 1 );

	std::string base_output_filename(argv[5]);

	//// Segment each frame independently
	cv::Size video_resolution = frame[0].size();

	size_t amount_of_vertices=0;
	size_t amount_of_edges=0;
	size_t amount_of_segments=0;
	std::vector< std::list<optical_flow_segment>> segments(frame.size());
	for(size_t v_t=0; v_t < frame.size(); ++v_t) {

		// Create a graph model
		std::list< int > vertices;
		std::list< edge_t > edges;

		for(int v_y=0; v_y < video_resolution.height; ++v_y) {
		for(int v_x=0; v_x < video_resolution.width; ++v_x) {

			int v_index = v_x + v_y*video_resolution.width;
			vertices.push_back(v_index);

			cv::Vec2f v_flow = flow[v_t].at<cv::Vec2f>(v_y, v_x);
			cv::Vec2i v_p(v_x, v_y);

			for(int x = v_x+1; x < std::min(video_resolution.width, v_x+1 + spatial_window_size/2); ++x) {
			
				cv::Vec2i p(x, v_y);

				float motion_similarity = l2_similarity(v_flow, flow[v_t].at<cv::Vec2f>(v_y,x));
				float dist_penalty = cv::norm(v_p - p, cv::NORM_L2);

				int index = x + v_y*video_resolution.width;
				edges.emplace_back(v_index, index, dist_penalty*motion_similarity);
			}
			for(int y = v_y+1; y < std::min(video_resolution.height, v_y+1 + spatial_window_size/2); ++y) {
			for(int x = std::max(0, v_x - spatial_window_size/2); x < std::min(video_resolution.width, v_x+1 + spatial_window_size/2); ++x) {
			
				cv::Vec2i p(x, y);

				float motion_similarity = l2_similarity(v_flow, flow[v_t].at<cv::Vec2f>(y,x));
				float dist_penalty = cv::norm(v_p - p, cv::NORM_L2);

				int index = x + y*video_resolution.width;
				edges.emplace_back(v_index, index, dist_penalty*motion_similarity);
			} } 
		} }

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

		std::set<int> parents;
		for(auto vertex=vertices.begin(); vertex != vertices.end(); ++vertex) {
			parents.insert(dsets.find_set(*vertex));
		}

		std::map<int, std::list<int>> parent2segment;
		std::set<int>::iterator p;
		int index;
		for(p=parents.begin(), index=0; p!=parents.end(); ++p, ++index) {
			parent2segment.insert(parent2segment.end(), std::pair<int, std::list<int>>(*p, std::list<int>()));
		}

		for(std::list<int>::iterator vertex=vertices.begin(); vertex != vertices.end(); ++vertex) {
			int i = dsets.find_set(*vertex);
			parent2segment[i].push_back(*vertex);
		}

		//
		for(std::map<int, std::list<int>>::iterator seg = parent2segment.begin(); seg != parent2segment.end(); ++seg) {

			int dx, dy;
			average_displacement(seg->second, flow[v_t], dx, dy);

			segments[v_t].emplace_back();
			segments[v_t].back()._avrg_dx = dx;
			segments[v_t].back()._avrg_dy = dy;
			segments[v_t].back()._mask.create(seg->second, video_resolution);
			segments[v_t].back()._points.splice(segments[v_t].back()._points.end(), seg->second);
		}

		amount_of_vertices += vertices.size();
		amount_of_edges += edges.size();
		amount_of_segments += segments[v_t].size();
       	}
	if(verbose) {
		std::cout << "Amount of vertices:\t" << amount_of_vertices << std::endl;
		std::cout << "Amount of edges:\t" << amount_of_edges << std::endl;
		std::cout << "Amount of segments:\t" << amount_of_segments << std::endl;
	}

	//// Link segments
	std::list<trajectory> complete_trajectories;
	std::list<trajectory> incomplete_trajectories;
	for(size_t t=0; t < segments.size(); ++t) {

		// for each trajectory
		for(std::list<trajectory>::iterator tr = incomplete_trajectories.begin(); tr != incomplete_trajectories.end(); ++tr) {
			const optical_flow_segment & b_seg = tr->back_segment();
			std::list<int> tracked_points(b_seg._points.begin(), b_seg._points.end());
			for(int & p : tracked_points) {
				p += b_seg._avrg_dx + b_seg._avrg_dy*video_resolution.width;
			}

			// find the best matching segment
			std::list<optical_flow_segment>::iterator best_seg;
			float max_similarity = 0;
			for(std::list<optical_flow_segment>::iterator seg = segments[t].begin(); seg != segments[t].end(); ++seg) {
				float similarity = Jaccard_similarity(seg->_points, tracked_points);
				if( similarity > max_similarity) {
					max_similarity = similarity;
					best_seg = seg;
				}
			}

			if(max_similarity > link_threshold) {	 // move segment
				tr->add_segment(*best_seg);
				tr++;
				segments[t].erase(best_seg);
			}
			else {					// keep segment
				complete_trajectories.splice(complete_trajectories.end(), incomplete_trajectories, tr++); 
			}
		}
		// the rest of segments become trajectories
		for(std::list<optical_flow_segment>::iterator seg = segments[t].begin(); seg != segments[t].end();) {
			incomplete_trajectories.push_back(trajectory(*seg,t));
			segments[t].erase(seg++);
		}
	}
	// remaining incomplete trajectories are supposed to be complited
	complete_trajectories.splice(complete_trajectories.end(), incomplete_trajectories);

	if(verbose) {
		std::cout << "Amount of trajectories: " << complete_trajectories.size() << std::endl;
	}

	//// Return output
	// for colouring of trajectories
	std::vector<int> random_numbers;
	unsigned int seed = rand() % 1000000;
	randomPermuteRange(pow(256,3), random_numbers, &seed); 

	std::vector<cv::Mat> output(amount_of_frames_for_seg);
	for(size_t i=0; i<output.size(); ++i) {
		cv::Mat temp = cv::Mat::zeros(video_resolution, CV_8UC3);
		temp.copyTo(output[i]);
	}

	// printing of trajectories
	int tr_id=0;
	for(std::list<trajectory>::iterator tr = complete_trajectories.begin(); tr != complete_trajectories.end(); ++tr, ++tr_id) {

		int t = tr->_start_frame;
		for(std::list<optical_flow_segment>::iterator seg = tr->_segments.begin(); seg != tr->_segments.end(); ++seg, ++t) {
			for(std::list<int>::iterator point=seg->_points.begin(); point != seg->_points.end(); ++point) {
				int y = (*point) / video_resolution.width;
				int x = (*point) % video_resolution.width;
			
				output[t].at<cv::Vec3b>(y,x) = colour_8UC3(random_numbers[tr_id]);
			}
		}
	}

	for(size_t i=0; i < output.size(); ++i) {
		std::string output_filename = base_output_filename + '_' + std::to_string(i) + ".png";
		cv::imwrite(output_filename, output[i]);
	}

	return 0;
}

void average_displacement(const std::list<int> & points, const cv::Mat & flow, int & avrg_dx, int & avrg_dy)
{
	int width = flow.size().width;

	float dx = 0;
	float dy = 0;

	for(std::list<int>::const_iterator p = points.begin(); p != points.end(); ++p) {
		int x = (*p) % width;
		int y = (*p) / width;

		const cv::Vec2i & flow_vec = flow.at<cv::Vec2f>(y,x);

		dx += flow_vec[0];
		dy += flow_vec[1];
	}

	avrg_dx = (int)(dx/points.size() + 0.5);
	avrg_dy = (int)(dy/points.size() + 0.5);
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

float Jaccard_similarity(const std::list<int> & lo, const std::list<int> & ro)
{
	std::unordered_set<int> points_union(lo.size() + ro.size());
	for(std::list<int>::const_iterator p=lo.begin(); p!=lo.end(); ++p) {
		points_union.insert(*p);
	}
	for(std::list<int>::const_iterator p=ro.begin(); p!=ro.end(); ++p) {
		points_union.insert(*p);
	}

	return (float)(lo.size() + ro.size() - points_union.size()) / (points_union.size());
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
