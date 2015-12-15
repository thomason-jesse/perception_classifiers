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
std::map<int, std::map<int, std::map<int, std::vector<perception_classifiers::Observations*>* > > > featuresCache;

bool g_caught_sigint=false;

void sig_handler(int sig){
	g_caught_sigint = true;
	ROS_INFO("caught sigint, cleared cache and starting= shutdown sequence...");

	// cleare featuresCache
	for (std::map<int, std::map<int, std::map<int, std::vector<perception_classifiers::Observations*>* > > >::iterator oiter = featuresCache.begin();
		 oiter != featuresCache.end(); ++oiter)
	{
		for (std::map<int, std::map<int, std::vector<perception_classifiers::Observations*>* > >::iterator biter = featuresCache[oiter->first].begin();
		 biter != featuresCache[oiter->first].end(); ++biter)
		{
			for (std::map<int, std::vector<perception_classifiers::Observations*>* >::iterator miter = featuresCache[oiter->first][biter->first].begin();
		 	 miter != featuresCache[oiter->first][biter->first].end(); ++miter)
			{
				for (int obs_idx=0; obs_idx < featuresCache[oiter->first][biter->first][miter->first]->size(); obs_idx++)
				{
					delete (*featuresCache[oiter->first][biter->first][miter->first])[obs_idx];
				}
				delete featuresCache[oiter->first][biter->first][miter->first];
			}
		}
	}

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
	std::vector<perception_classifiers::Observations*>* observations =
						new std::vector<perception_classifiers::Observations*>();
	int object = req.object;
	int behavior = req.behavior;
	int modal = req.modality;
	std::string filepath = fp_data + object_base + boost::lexical_cast<std::string>(object) + "/" 
			+ behaviorList[behavior] + "/" + modalList[modal] + "/" + filename;
	std::ifstream file(filepath.c_str());

	// if in cache, just return that instead of doing file read every time
	if (featuresCache.count(object) == 1 && featuresCache[object].count(behavior) == 1 &&
		featuresCache[object][behavior].count(modal) == 1)
	{
		for (int obs_idx=0; obs_idx < featuresCache[object][behavior][modal]->size(); obs_idx++)
		{
			res.rows.push_back((*(*featuresCache[object][behavior][modal])[obs_idx]));
		}
		return true;
	}
	
	if(file.fail()){
		ROS_ERROR("File doesn't exist due to invalid arguments. Attempted to open %s", filepath.c_str());
	} else{
		ROS_DEBUG("Opened features file.");
		/* We make the hard assumption (for now) that if there is a next line, there are 5 additional lines
		 */
		int lineNum = 0;
		while(!file.eof()){
			perception_classifiers::Observations* o = new perception_classifiers::Observations();
			o->features = getNextLineAndSplit(file);
			if(o->features.size() > 0)							//catches the last vector of the file, which is empty.
			{
				observations->push_back(o);
				res.rows.push_back(*o);
			}
		}
		featuresCache[object][behavior][modal] = observations;
		return true;
	}
	return false;
}

int main(int argc, char **argv){
	ros::init(argc, argv, "fetch_feature_node");
	ros::NodeHandle n;
	ros::ServiceServer srv = n.advertiseService("fetch_feature_service", service_cb);

	// TODO: should be read from config.txt using method shared with classifier_services
	behaviorList +=  "look";
	modalList += "shape", "color", "fc7";

	// set shutdown procedure call
  	signal(SIGINT, sig_handler);

	ros::Rate r(5);
	while(ros::ok()){
		ros::spinOnce();
		r.sleep();
	}
	return 0;
}