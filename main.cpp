#include <bits/stdc++.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/file_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/distances.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>

std::set<std::pair<int, int>> greedyTriangulationPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double R) {
	// Normal Estimation
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setKSearch (20);
	n.compute (*normals);
	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals
	// 
	// Create search tree* (for points with normals)
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud_with_normals);
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;
	// Set the maximum distance between connected points (edge length)
	gp3.setSearchRadius(R);
	// Set typical values for the parameters
	gp3.setMu (2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(false);
	// Final results
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);

	std::set<std::pair<int, int>> edges;
	for(auto tri : triangles.polygons) {
		edges.insert({ tri.vertices[0], tri.vertices[1] });
		edges.insert({ tri.vertices[1], tri.vertices[2] });
		edges.insert({ tri.vertices[2], tri.vertices[0] });
	}
	return edges;
}

void createEdges(pcl::PointCloud<pcl::PointXYZ>::Ptr &mesh, std::set<std::pair<int, int>> &m, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
	for(auto e : m) {
		auto p1 = cloud->at(e.first);
		auto p2 = cloud->at(e.second);
		double dirX = p2.x - p1.x;
		double dirY = p2.y - p1.y;
		double dirZ = p2.z - p1.z;
		for(int i = 0; i <= 100; i++) {
			pcl::PointXYZ p;
			p.x = p1.x + i*dirX/100;
			p.y = p1.y + i*dirY/100;
			p.z = p1.z + i*dirZ/100;
			mesh->push_back(p);
		}
	}
}

std::set<std::pair<int, int>> myAlgo(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double R) {
	// set of edges (return value)
	std::set<std::pair<int, int>> edges;

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setRadiusSearch(R);
	n.compute (*normals);
	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals

	// For center and radius of circles
	pcl::SampleConsensusModelCircle3D<pcl::PointNormal> pseg(cloud_with_normals);
	pcl::search::KdTree<pcl::PointNormal> kdtree;
	kdtree.setInputCloud(cloud_with_normals);
	
	//First is for nnn of pts, second for finding the pts in circumcircle
	std::vector <int> indx, id2;
	std::vector <float> sqDist, sq2;
	for(int i = 0; i < (int)cloud_with_normals->size(); i++) {
		indx.clear(); id2.clear();
		sqDist.clear(); sq2.clear();
		kdtree.radiusSearch(cloud_with_normals->at(i), R, indx, sqDist);
		for(int m = 0; m < indx.size(); m++) {
			if(pcl::euclideanDistance(cloud_with_normals->at(i), cloud_with_normals->at(indx[m])) <= 1e-2) {
				continue;
			}
			for(int n = 0; n < m; n++) {
				if(pcl::euclideanDistance(cloud_with_normals->at(i), cloud_with_normals->at(indx[n])) <= 1e-2) {
					continue;
				}
				std::vector <int> indices = {i, indx[m], indx[n]};
				Eigen::VectorXf modelCoeffs;
				pseg.computeModelCoefficients(indices, modelCoeffs);
				if(modelCoeffs[0]!=modelCoeffs[0] || modelCoeffs[1]!=modelCoeffs[1] || modelCoeffs[2]!=modelCoeffs[2] || modelCoeffs[3]!=modelCoeffs[3]) {
					continue;
				}
				pcl::PointNormal center(modelCoeffs[0], modelCoeffs[1], modelCoeffs[2], modelCoeffs[4], modelCoeffs[5], modelCoeffs[6]);
				//Set of pts within radius from circumcenter
				std::set<int> indx_set;
				auto nnn = kdtree.radiusSearch(center, modelCoeffs[3]-R/100, id2, sq2);
				for(auto ix : id2) {
					indx_set.insert(ix);
				}
				indx_set.erase(i);
				indx_set.erase(indx[m]);
				indx_set.erase(indx[n]);

				// Checking angle between local normal and planes normal
				Eigen::Vector3f faceNormal(modelCoeffs[4], modelCoeffs[5], modelCoeffs[6]);
				Eigen::Vector3f localNormal(cloud_with_normals->at(i).getNormalVector3fMap());
				faceNormal.normalize();
				localNormal.normalize();
				double cosine = faceNormal.dot(localNormal);

				if(indx_set.empty() && !(std::abs(cosine) < std::cos(M_PI/36))) {
					edges.insert({ i, indx[m] });
					edges.insert({ indx[m], indx[n] });
					edges.insert({ indx[n], i });
				}
			}
		}
	}
	return edges;
}

int main() {
	// Radius Search
	double R;
	std::cout<<"Enter radius to search R:\n";
	std::cin>>R;
	
	//filename
	std::string filename;
	std::cout<<"Enter the cloud filename\n";
	std::cin>>filename;
	std::fstream file;
	file.open(filename, std::ios::in);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if(!file.is_open()) {
		std::cout<<"Error opening file! Creating random cloud"<<std::endl;
		return -1;
	}
	while(!file.eof()) {
		double x, y, z;
		file>>x>>y>>z;
		pcl::PointXYZ p(x, y, z);
		cloud->push_back(p);
	}
	
	pcl::visualization::PCLVisualizer view("Initial cloud");
	view.addPointCloud(cloud, "cloud");
	while(!view.wasStopped()) {
		view.spinOnce();
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloudPCL(new pcl::PointCloud<pcl::PointXYZ>), 
										meshCloudMY(new pcl::PointCloud<pcl::PointXYZ>);

	auto PCLset = greedyTriangulationPCL(cloud, R);
	createEdges(meshCloudPCL, PCLset, cloud);
	pcl::visualization::PCLVisualizer view1("PCL mesh");
	view1.addPointCloud(meshCloudPCL, "meshPCL");
	view1.addPointCloud(cloud, "cloud");
	while(!view1.wasStopped()) {
		view1.spinOnce();
	}

	auto MYset = myAlgo(cloud, R);
	createEdges(meshCloudMY, MYset, cloud);
	pcl::visualization::PCLVisualizer view2("My mesh");
	view2.addPointCloud(meshCloudMY, "meshCloudMY");
	view2.addPointCloud(cloud, "cloud");
	while(!view2.wasStopped()) {
		view2.spinOnce();
	}
	return 0;
}