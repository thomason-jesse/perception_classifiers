#include <ros/ros.h>
#include <ros/package.h>
#include <perception_classifiers/Observations.h>
#include <perception_classifiers/FetchFeatures.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <signal.h> 
#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>
using namespace boost::assign;

/*
 *  Some constants for the location of feature data.
 *  Currently use fake data for what I image the file structure will be.
 */
std::string fp_data 					= ros::package::getPath("perception_classifiers") + "/data/";
std::string filename					= "features.csv";
std::string object_base 				= "obj";
std::vector<std::string> behaviorList;
std::vector<std::string> modalList;

bool g_caught_sigint=false;

void sig_handler(int sig){
	g_caught_sigint = true;
	ROS_INFO("caught sigint, init shutdown sequence...");
	ros::shutdown();
	exit(1);
};

std::vector<float> getNextLineAndSplit(std::istream& str){
	std::vector<float>			result;
	std::string					line;
	std::getline(str,line);

	std::stringstream			lineStream(line);
	std::string					cell;
	bool						firstTime = true;
	
	while(std::getline(lineStream,cell,',')){
		if(!firstTime)												//skips the first cell, which is headers
			result.push_back(boost::lexical_cast<float>(cell));
		else
			firstTime = false;
	}
	return result;
}

bool service_cb(perception_classifiers::FetchFeatures::Request &req, perception_classifiers::FetchFeatures::Response &res){
	std::vector<perception_classifiers::Observations> observations;
	int object = req.object;
	int behavior = req.behavior;
	int modal = req.modality;
	std::string filepath = fp_data + object_base + boost::lexical_cast<std::string>(object) + "/" 
			+ behaviorList[behavior] + "/" + modalList[modal] + "/" + filename;
	std::ifstream file(filepath.c_str());
	
	if(file.fail()){
		ROS_ERROR("File doesn't exist due to invalid arguments. Attempted to open %s", filepath.c_str());
	} else{
		ROS_DEBUG("Opened features file.");
		/* We make the hard assumption (for now) that if there is a next line, there are 5 additional lines
		 */
		int lineNum = 0;
		while(!file.eof()){
			perception_classifiers::Observations o;
			o.features = getNextLineAndSplit(file);
			if(o.features.size() > 0)							//catches the last vector of the file, which is empty.
				observations.push_back(o);
		}
		res.rows = observations;
		return true;
	}
	res.rows = observations;
	return false;
}

int main(int argc, char **argv){
	ros::init(argc, argv, "fetch_feature_node");
	ros::NodeHandle n;
	ros::ServiceServer srv = n.advertiseService("fetch_feature_service", service_cb);

	// TODO: should be read from config.txt using method shared with classifier_services
	behaviorList +=  "look";
	modalList += "shape", "color", "fc7";

	ros::Rate r(5);
	while(ros::ok()){
		ros::spinOnce();
		r.sleep();
	}
	return 0;
}