#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/file_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/distances.h>
#include <Eigen/Dense>

int main() {
	std::string filename;
	std::cout<<"Enter the cloud filename\n";
	std::cin>>filename;
	std::fstream file;
	file.open(filename, std::ios::in);
	if(!file.is_open()) {
		std::cout<<"Error opening file! Please try again."<<std::endl;
		return -1;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	while(!file.eof()) {
		double x, y, z;
		file>>x>>y>>z;
		pcl::PointXYZ p(x, y, z);
		cloud->push_back(p);
	}
	// std::map<int, std::set <int>> mesh;
	// std::set<std::tuple<int, int, int>> mesh;
	std::set<std::set<int>> mesh;
	pcl::SampleConsensusModelCircle3D<pcl::PointXYZ> pseg(cloud);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	std::vector <int> indx, id2;
	std::vector <float> sqDist, sq2;
	for(int i = 0; i < (int)cloud->size(); i++) {
		indx.clear(); id2.clear();
		sqDist.clear(); sq2.clear();
		kdtree.nearestKSearch(cloud->at(i), 30, indx, sqDist);
		for(int m = 0; m < indx.size(); m++) {
			if(pcl::euclideanDistance(cloud->at(i), cloud->at(indx[m])) <= 1e-2) {
				continue;
			}
			for(int n = 0; n < m; n++) {
				if(pcl::euclideanDistance(cloud->at(i), cloud->at(indx[n])) <= 1e-2) {
					continue;
				}
				std::vector <int> indices = {i, indx[m], indx[n]};
				Eigen::VectorXf modelCoeffs;
				pseg.computeModelCoefficients(indices, modelCoeffs);
				if(modelCoeffs[0]!=modelCoeffs[0] || modelCoeffs[1]!=modelCoeffs[1] || modelCoeffs[2]!=modelCoeffs[2] || modelCoeffs[3]!=modelCoeffs[3]) {
					continue;
				}
				pcl::PointXYZ center(modelCoeffs[0], modelCoeffs[1], modelCoeffs[2]);
				std::cout<<i<<", "<<indx[m]<<", "<<indx[n]<<"\n";
				std::cout<<cloud->at(i)<<", "<<cloud->at(indx[m])<<", "<<cloud->at(indx[n])<<"\n";
				// std::cout<<modelCoeffs<<std::endl;
				std::set<int> indx_set;
				auto nnn = kdtree.radiusSearch(center, modelCoeffs[3], id2, sq2);
				// std::cout<<nnn<<std::endl;
				std::cout<<"*********************\n";
				for(auto ix : id2) {
					indx_set.insert(ix);
				}
				indx_set.erase(i);
				indx_set.erase(indx[m]);
				indx_set.erase(indx[n]);
				if(indx_set.empty()) {
					std::set<int> A;
					A.insert(i);
					A.insert(indx[m]);
					A.insert(indx[n]);
					mesh.insert(A);
				}
				// if(nnn == 0) {
					// mesh[i].insert(indx[m]);
					// mesh[i].insert(indx[n]);
					// mesh[m].insert(i);
					// mesh[m].insert(indx[n]);
					// mesh[n].insert(indx[m]);
					// mesh[n].insert(i);
					// std::tuple<int, int, int> A = {i, indx[m], indx[n]};
					// std::set<int> A;
					// A.insert(i);
					// A.insert(indx[m]);
					// A.insert(indx[n]);
					// mesh.insert(A);
				// } 

			}
		}
	}
	for(auto adj : mesh) {
		for(auto p : adj) {
			std::cout<<p<<", ";
		}
		std::cout<<"\n";
		// std::cout<<std::get<0>(adj)<<", "<<std::get<1>(adj)<<", "<<std::get<2>(adj)<<std::endl;
		// std::cout<<"for "<<adj.first<<": ";
		// for(auto neighbr : adj.second) {
		// 	std::cout<<neighbr<<", ";
		// }
		// std::cout<<std::endl;
	}
	return 0;
}