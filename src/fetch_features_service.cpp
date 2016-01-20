#include <ros/ros.h>
#include <ros/package.h>
#include <perception_classifiers/Observations.h>
#include <perception_classifiers/FetchFeatures.h>
#include <perception_classifiers/FetchAllFeatures.h>
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
std::string condition;
std::string config_fn;

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
			std::vector<int> miter_ids;
			for (std::map<int, std::vector<perception_classifiers::Observations*>* >::iterator miter = featuresCache[oiter->first][biter->first].begin();
		 	 miter != featuresCache[oiter->first][biter->first].end(); ++miter)
			{
				for (int obs_idx=0; obs_idx < featuresCache[oiter->first][biter->first][miter->first]->size(); obs_idx++)
				{
					delete (*featuresCache[oiter->first][biter->first][miter->first])[obs_idx];
				}
				miter_ids.push_back(miter->first);
			}
			for (int midx=0; midx < miter_ids.size(); midx++)
			{
				delete featuresCache[oiter->first][biter->first][midx];
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
	
	if(file.fail())
	{
		if (!req.allow_missing)
		{
			ROS_ERROR("File doesn't exist due to invalid arguments. Attempted to open %s", filepath.c_str());
		}
		else
		{
			perception_classifiers::Observations* o = new perception_classifiers::Observations();
			std::vector<float> blank_f;
			o->features = blank_f;
			observations->push_back(o);
			res.rows.push_back(*o);
			featuresCache[object][behavior][modal] = observations;
			return true;
		}
	} 
	else
	{
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

bool get_all_features_service(perception_classifiers::FetchAllFeatures::Request &req, perception_classifiers::FetchAllFeatures::Response &res)
{
	int object = req.object;
	std::vector<float> features;

	for (int b_idx=0; b_idx < behaviorList.size(); b_idx++)
	{
		for (int m_idx=0; m_idx < modalList.size(); m_idx++)
		{
			perception_classifiers::FetchFeatures ff;
			ff.request.object = object;
			ff.request.behavior = b_idx;
			ff.request.modality = m_idx;
			ff.request.allow_missing = true;
			bool r = service_cb(ff.request, ff.response);
			if (r == false)
				return false;
			// average over the observations in this context to keep vector fixed-length
			for (int f=0; f < ff.response.rows[0].features.size(); f++)
			{
				float f_sum = 0;
				for (int obs_idx=0; obs_idx < ff.response.rows.size(); obs_idx++)
				{
					f_sum += ff.response.rows[obs_idx].features[f];
				}
				features.push_back( f_sum / ff.response.rows.size() );
			}
		}
	}

	res.features = features;
	return true;
}

int main(int argc, char **argv)
{

	condition = argv[1];
	config_fn = ros::package::getPath("perception_classifiers") + "/" + condition +".config";

	ros::init(argc, argv, "fetch_feature_node");
	ros::NodeHandle n;
	ros::ServiceServer fetch_feature_service = n.advertiseService("fetch_feature_service", service_cb);
	ros::ServiceServer fetch_all_features_service = n.advertiseService("fetch_all_features_service", get_all_features_service);

	// read config file to get behaviors and modalities
	// config file format: CSV with first line names of modalities, subsequent lines behavior names
	// followed by list of features in behavior/modality combination, 0 if no classifier in combo
	std::ifstream infile(config_fn.c_str());
	if (infile.fail())
	{
		ROS_ERROR("missing config file at %s", config_fn.c_str());
	}
	bool first_line = true;
	std::cout << "reading " << config_fn.c_str() << "\n"; // debug
	while (!infile.eof())
	{
		std::string line;
		std::string entry;
		getline(infile, line);
		if (first_line)
		{
			first_line = false;
			std::istringstream ss(line);
			bool blank_entry = true;
			while (getline(ss, entry, ','))
			{
				if (blank_entry == true)
				{
					blank_entry = false;
					continue;
				}
				modalList.push_back(entry.c_str());
				std::cout << "pushed modality " << entry.c_str() << "\n";  // DEBUG
			}
		}
		else
		{
			std::istringstream ss(line);
			getline(ss, entry, ',');
			behaviorList.push_back(entry.c_str());
			std::cout << "pushed behavior " << entry.c_str() << "\n";  // DEBUG
		}
	}

	// set shutdown procedure call
  	signal(SIGINT, sig_handler);

	ros::Rate r(5);
	while(ros::ok()){
		ros::spinOnce();
		r.sleep();
	}
	return 0;
}
