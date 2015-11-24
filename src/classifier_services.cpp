#include "perception_classifiers/getFreeClassifierID.h"
#include "perception_classifiers/loadClassifiers.h"
#include "perception_classifiers/runClassifier.h"
#include "perception_classifiers/trainClassifier.h"

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/lexical_cast.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Empty.h>

#include <map>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

bool getFreeClassifierID(perception_classifiers::getFreeClassifierID::Request &req,
				        perception_classifiers::getFreeClassifierID::Response &res);

bool loadClassifiers(perception_classifiers::loadClassifiers::Request &req,
				     perception_classifiers::loadClassifiers::Response &res);

bool saveClassifiers(std_srvs::Empty::Request &,
					 std_srvs::Empty::Response &);

bool deleteClassifiers(std_srvs::Empty::Request &,
					   std_srvs::Empty::Response &);

bool runClassifier(perception_classifiers::runClassifier::Request &req,
				   perception_classifiers::runClassifier::Response &res);

bool trainClassifier(perception_classifiers::trainClassifier::Request &req,
				     perception_classifiers::trainClassifier::Response &res);

// variables related to feature space, to be read in from configuration file
int max_classifier_ID;
vector<int> classifier_IDs;
int num_behaviors;
int num_modalities;
vector< vector<int> > num_features;

// variables related to classifiers
map<int, vector< vector<CvSVM*> > > classifiers;
map<int, vector< vector<double> > > confidences;

// initialize node and offer services
int main(int argc, char **argv)
{
	ros::init(argc, argv, "classifier_services");
  	ros::NodeHandle n;

  	ros::ServiceServer get_free_classifier_ID = n.advertiseService("get_free_classifier_ID", getFreeClassifierID);
	ros::ServiceServer load_classifiers = n.advertiseService("load_classifiers", loadClassifiers);
	ros::ServiceServer save_classifiers = n.advertiseService("save_classifiers", saveClassifiers);
	ros::ServiceServer delete_classifiers = n.advertiseService("delete_classifiers", deleteClassifiers);
	ros::ServiceServer run_classifier = n.advertiseService("run_classifier", runClassifier);
	ros::ServiceServer train_classifier = n.advertiseService("train_classifier", trainClassifier);
	ros::spin();

	// read config file to find number of behaviors and modalities and populate num_features matrix
	// config file format: CSV with first line names of modalities, subsequent lines behavior names
	// followed by list of features in behavior/modality combination, 0 if no classifier in combo
	ifstream infile("config.txt");
	bool first_line = true;
	while (infile)
	{
		string line;
		if (!getline(infile, line) or first_line)
		{
			first_line = false;
			break;
		}
		istringstream ss(line);
		bool first_entry = true;
		vector<int> num_features_for_behaviors;
		while (ss)
		{
			if (first_entry)
			{
				first_entry = false;
				break;
			}
			string entry;
			if (!getline(ss, entry, ','))
				break;
			num_features_for_behaviors.push_back( atoi(entry.c_str()) );
		}
		num_features.push_back(num_features_for_behaviors);
	}
	num_behaviors = num_features.size();
	num_modalities = num_features[0].size();
	if (!infile.eof())
	{
		cerr << "missing config.txt file\n";
	}
	max_classifier_ID = 0;

	return 0;
}

// return the maximum classifier ID known
bool getFreeClassifierID(perception_classifiers::getFreeClassifierID::Request &req,
				        perception_classifiers::getFreeClassifierID::Response &res)
{
	res.ID = max_classifier_ID+1;
	max_classifier_ID += 1;
	return true;
}

// read classifiers and classifier confidences out of files
bool loadClassifiers(perception_classifiers::loadClassifiers::Request &req,
				     perception_classifiers::loadClassifiers::Response &res)
{
	classifier_IDs.clear();

	// read classifier confidences in from file
	ifstream infile("confidences.csv");
	while (infile)
	{
		string line;
		if (!getline(infile, line))
			break;
		istringstream ss(line);
		string id;
		getline(ss, id, 's');
		int classifier_ID = atoi(id.c_str());
		if (classifier_ID > max_classifier_ID)
			max_classifier_ID = classifier_ID;
		classifier_IDs.push_back(classifier_ID);
		vector< vector<double> > confidences_for_behaviors;
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			vector<double> confidences_for_modalities;
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				string conf_str;
				getline(ss, conf_str, ',');
				confidences_for_modalities.push_back( atoi(conf_str.c_str()) );
			}
			confidences_for_behaviors.push_back(confidences_for_modalities);
		}
		confidences[classifier_ID] = confidences_for_behaviors;
	}

	for (int idx=0; idx < classifier_IDs.size(); idx++)
	{
		//read classifier from file
		vector< vector<CvSVM*> > sub_c;
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			vector<CvSVM*> m_c;
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				if (num_features[b_idx][m_idx] == 0)
					continue;
				ostringstream fn;
				fn << "classifier" << classifier_IDs[idx]
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				CvSVM c;
				c.load(fn.str().c_str());
				m_c.push_back(&c);
			}
			sub_c.push_back(m_c);
		}
		classifiers[classifier_IDs[idx]] = sub_c;
	}

	res.success = true;
	return true;
}

// write classifiers and classifier confidences out to files which can later
// be loaded with loadClassifiers
bool saveClassifiers(std_srvs::Empty::Request &,
					 std_srvs::Empty::Response &)
{
	ofstream conf_file;
	conf_file.open("confidences.csv");

	bool first_line = true;
	for (map<int, vector< vector<CvSVM*> > >::iterator iter = classifiers.begin();
		 iter != classifiers.end(); ++iter)
	{
		conf_file << boost::lexical_cast<string>(iter->first);  // confidences headed by classifier ID
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				// write classifier confidences out to file
				conf_file << ',';
				conf_file << boost::lexical_cast<string>(confidences[iter->first][b_idx][m_idx]);

				if (num_features[b_idx][m_idx] == 0)
					continue;

				// write classifier out to file
				ostringstream fn;
				fn << "classifier" << iter->first
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				classifiers[iter->first][b_idx][m_idx]->save(fn.str().c_str());
			}
		}
		if (!first_line)
			conf_file << '\n';
		first_line = false;
	}

	conf_file.close();

	return true;
}

// delete classifier and confidence files from disk
bool deleteClassifiers(std_srvs::Empty::Request &,
					   std_srvs::Empty::Response &)
{
	for (map<int, vector< vector<CvSVM*> > >::iterator iter = classifiers.begin();
		 iter != classifiers.end(); ++iter)
	{
		//delete all subclassifier files
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				if (num_features[b_idx][m_idx] == 0)
					continue;
				ostringstream fn;
				fn << "classifier" << iter->first
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				remove(fn.str().c_str());
			}
		}

		// delete classifier confidences file
		remove("confidences.csv");
	}

	return true;
}

// run a specified classifier on a vector of objects and report results and confidences
bool runClassifier(perception_classifiers::runClassifier::Request &req,
				     perception_classifiers::runClassifier::Response &res)
{
	//run classifier in each relevant behavior, modality combination 
	double decision = 0;
	int sub_classifiers_used = 0;
	vector<double> _dec;
	for (int b_idx=0; b_idx < num_behaviors; b_idx++)
	{
		for (int m_idx=0; m_idx < num_modalities; m_idx++)
		{
			if (num_features[b_idx][m_idx] == 0)
				_dec.push_back(0);
				continue;
			sub_classifiers_used += 1;

			// access feature-getting service and use it to populate rows of test matrix
			Mat test_data;
			vector< vector<double> > observations;
			vector<double> temp_null_ob;
			temp_null_ob.push_back(0.0);
			observations.push_back(temp_null_ob);
			//TODO: initialize observations by some feature getting service (req.object_ID, b_idx, o_idx)
			for (int obs_idx=0; obs_idx < observations.size(); obs_idx++)
			{
				Mat observation(observations.size(), num_features[b_idx][m_idx],
								DataType<double>::type, &observations[obs_idx]);
				test_data.push_back(observation);
			}

			// run classifier on each observation
			double num_positive = 0;
			for (int obs_idx=0; obs_idx < observations.size(); obs_idx++)
			{
				double response = classifiers[req.classifier_ID][b_idx][m_idx]
								 ->predict(test_data.row(obs_idx));
				if (response == 1)
					num_positive += 1.0;
			}

			// average observation decisions to decide this sub classifier's decision
			// could instead do majority voting
			double _decision = 2*(num_positive / observations.size()) - 1;
			_dec.push_back(_decision);

			// add to overall decision with confidence weight
			decision += _decision * confidences[req.classifier_ID][b_idx][m_idx];
		}
	}

	//set return value based on decision score
	if (decision > 0)
		res.result = 1;
	else
		res.result = -1;
	if (sub_classifiers_used > 0)
		res.confidence = abs(decision / sub_classifiers_used);
	else
		res.confidence = 0;
	res.sub_classifier_decisions = _dec;

	return true;
}

// train a classifier with given object IDs and labels and store it under the given classifier ID
bool trainClassifier(perception_classifiers::trainClassifier::Request &req,
				     perception_classifiers::trainClassifier::Response &res)
{
	int classifier_ID = req.classifier_ID;
	vector<int> object_IDs = req.object_IDs;
	int num_objects = static_cast<int>(object_IDs.size());
	if (std::find(classifier_IDs.begin(), classifier_IDs.end(), classifier_ID) == classifier_IDs.end())
	{
		classifier_IDs.push_back(classifier_ID);
		if (classifier_ID > max_classifier_ID)
			max_classifier_ID = classifier_ID;
	}

	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// for each behavior and modality, retrieve relevant features for each object and train sub-classifiers
	vector< vector<CvSVM*> > sub_classifiers;
	vector< vector<double> > sub_confidence;
	for (int b_idx=0; b_idx < num_behaviors; b_idx++)
	{
		vector<CvSVM*> modality_classifiers;
		vector<double> modality_confidence;
		for (int m_idx=0; m_idx < num_modalities; m_idx++)
		{
			//if there are no features for this combination, don't create classifier
			if (num_features[b_idx][m_idx] == 0)
			{
				modality_classifiers.push_back(NULL);
				modality_confidence.push_back(0);
				continue;
			}

			// retrieve relevant features for each object
			vector<int> num_observations;
			Mat train_data;
			Mat responses;
			for (int o_idx=0; o_idx < num_objects; o_idx++)
			{
				// access feature-getting service and use it to populate a row of train_data matrix
				vector< vector<double> > observations;
				vector<double> temp_null_ob;
				temp_null_ob.push_back(0.0);
				observations.push_back(temp_null_ob);
				// TODO: initialize observations through some feature getting service (o_idx, b_idx, m_idx)
				num_observations.push_back(observations.size());
				for (int obs_idx=0; obs_idx < observations.size(); obs_idx++)
				{
					Mat observation(observations.size(), num_features[b_idx][m_idx],
									DataType<double>::type, &observations[obs_idx]);
					train_data.push_back(observation);
					if (req.positive_example[o_idx] == true)
						responses.push_back(1);
					else
						responses.push_back(-1);
				}
			}

			// do leave-one-out cross validation to determine confidence in this classifier
			CvSVM c;
			double x_fold_correct = 0;
			for (int fo_idx=0; fo_idx < num_objects; fo_idx++)
			{
				Mat train_fold;
				Mat responses_fold;
				int read_from_row = 0;
				int fold_rows_begin = -1;
				for (int to_idx=0; to_idx < num_objects; to_idx++)
				{
					if (fo_idx != to_idx)
					{
						for (int obs_idx=0; obs_idx < num_observations[to_idx]; obs_idx++)
						{
							train_fold.push_back(train_data.row(read_from_row+obs_idx));
							responses_fold.push_back(responses.row(read_from_row+obs_idx));
						}
					}
					else
						fold_rows_begin = read_from_row;
					read_from_row += num_observations[to_idx];
				}

				c.train(train_fold, responses_fold, Mat(), Mat(), params);
				double observations_correct = 0;
				for (int obs_idx=0; obs_idx < num_observations[fo_idx]; obs_idx++)
				{
					double response = c.predict(train_data.row(fold_rows_begin+obs_idx));
					if (response == responses.at<float>(fo_idx, 0))
						observations_correct += 1.0;
				}
				x_fold_correct += observations_correct / num_observations[fo_idx];
			}
			double confidence = x_fold_correct / static_cast<float>(num_objects);
			modality_confidence.push_back(confidence);

			// train classifier with all gathered data and store in structure
			c.train(train_data, responses, Mat(), Mat(), params);
			modality_classifiers.push_back(&c);

		}

		sub_classifiers.push_back(modality_classifiers);
		sub_confidence.push_back(modality_confidence);
	}
	classifiers[classifier_ID] = sub_classifiers;
	confidences[classifier_ID] = sub_confidence;

	res.success = true;
	return true;
}

