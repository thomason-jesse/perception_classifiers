#include "perception_classifiers/getFreeClassifierID.h"
#include "perception_classifiers/loadClassifiers.h"
#include "perception_classifiers/runClassifier.h"
#include "perception_classifiers/trainClassifier.h"
#include "perception_classifiers/FetchFeatures.h"

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/lexical_cast.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Empty.h>

#include <signal.h>
#include <sys/stat.h>

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

// filepath data
string config_fn = ros::package::getPath("perception_classifiers") + "/config.txt";
string class_fn = ros::package::getPath("perception_classifiers") + "/classifiers/";
string conf_fn = ros::package::getPath("perception_classifiers") + "/classifiers/confidences.csv";

// variables related to feature space, to be read in from configuration file
int max_classifier_ID;
vector<int> classifier_IDs;
int num_behaviors;
int num_modalities;
vector< vector<int> > num_features;

// variables related to classifiers
map<int, vector< vector<CvSVM*> > > classifiers;
map<int, vector< vector<float> > > confidences;

// services we will call
ros::ServiceClient fetch_features;

// caches
struct cache_response
{
	int result;
	float confidence;
	vector<float>* sub_classifier_decisions;
};
map<int, map<int, cache_response*> > run_classifier_cache;

// calculate kappa statistic
float kappa(int cm[2][2])
{
	double sr[2];
    double sc[2];
    double sw = 0.0;
    for (int i = 0; i < 2; i++)
    {
		for (int j = 0; j < 2; j++)
		{
			sr[i] += cm[i][j];
			sc[j] += cm[i][j];
			sw += cm[i][j];
		}
    }
    if (sw == 0)
    	return 0;
    double po = 0.0;
    double pe = 0.0;
    for (int i = 0; i < 2; i++)
    {
		pe += (sr[i] * sc[i]);
		po += cm[i][i];
    }
    pe /= (sw*sw);
    po /= sw;

    if (pe < 1.0)
		return (po - pe) / (1.0 - pe);
    else
		return 1.0;
}

void freeClassifierCache(int cid)
{
	for (map<int, cache_response*>::iterator oiter = run_classifier_cache[cid].begin();
	 oiter != run_classifier_cache[cid].end(); ++oiter)
	{
		if (run_classifier_cache[cid][oiter->first] != NULL)
		{
			delete run_classifier_cache[cid][oiter->first]->sub_classifier_decisions;
			delete run_classifier_cache[cid][oiter->first];
			run_classifier_cache[cid][oiter->first] = NULL;
		}
	}
}

void freeClassifierMemory()
{
	// delete classifiers from memory
	for (map<int, vector< vector<CvSVM*> > >::iterator iter = classifiers.begin();
		 iter != classifiers.end(); ++iter)
	{
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				if (classifiers[iter->first][b_idx][m_idx] != NULL)
				{
					delete classifiers[iter->first][b_idx][m_idx];
				}
			}
		}
	}

	// delete run classifier cache
	for (map<int, map<int, cache_response*> >::iterator citer = run_classifier_cache.begin();
		 citer != run_classifier_cache.end(); ++citer)
	{
		freeClassifierCache(citer->first);
	}
}

void customShutdown(int sig)
{
	ROS_INFO("caught sigint, freeing memory and starting shutdown sequence...");
	freeClassifierMemory();
	ros::shutdown();
}

// initialize node and offer services
int main(int argc, char **argv)
{
	ros::init(argc, argv, "classifier_services");
  	ros::NodeHandle n;

  	// set shutdown procedure call
  	signal(SIGINT, customShutdown);

  	// advertise own services 
  	ros::ServiceServer get_free_classifier_ID = n.advertiseService("get_free_classifier_ID", getFreeClassifierID);
	ros::ServiceServer load_classifiers = n.advertiseService("load_classifiers", loadClassifiers);
	ros::ServiceServer save_classifiers = n.advertiseService("save_classifiers", saveClassifiers);
	ros::ServiceServer delete_classifiers = n.advertiseService("delete_classifiers", deleteClassifiers);
	ros::ServiceServer run_classifier = n.advertiseService("run_classifier", runClassifier);
	ros::ServiceServer train_classifier = n.advertiseService("train_classifier", trainClassifier);

	// connect to helper services
	fetch_features = n.serviceClient<perception_classifiers::FetchFeatures>("fetch_feature_service");

	// read config file to find number of behaviors and modalities and populate num_features matrix
	// config file format: CSV with first line names of modalities, subsequent lines behavior names
	// followed by list of features in behavior/modality combination, 0 if no classifier in combo
	ifstream infile(config_fn.c_str());
	if (infile.fail())
	{
		ROS_ERROR("missing config file at %s", config_fn.c_str());
	}
	bool first_line = true;
	cout << "reading " << config_fn.c_str() << "\n"; // debug
	while (!infile.eof())
	{
		string line;
		getline(infile, line);
		if (first_line)
		{
			first_line = false;
			continue;
		}
		istringstream ss(line);
		bool first_entry = true;
		string entry;
		vector<int> num_features_for_behaviors;
		while (getline(ss, entry, ','))
		{
			if (!first_entry)
			{
				num_features_for_behaviors.push_back( atoi(entry.c_str()) );
			}
			else
			{
				first_entry = false;
			}
		}
		num_features.push_back(num_features_for_behaviors);
	}
	num_behaviors = num_features.size();
	num_modalities = num_features[0].size();
	cout << "...num_behaviors: " << num_behaviors << ", num_modalities: " << num_modalities << "\n"; // debug
	max_classifier_ID = 0;

	ros::spin();

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
	// debug
	cout << "loadClassifiers called\n";

	if (max_classifier_ID > 0)
	{
		cout << "... ignoring directive since some classifiers are already in use\n";
		res.success = true;
		return true;
	}

	// read classifier confidences in from file
	ifstream infile(conf_fn.c_str());
	while (infile)
	{
		string line;
		if (!getline(infile, line))
			break;
		if (line.length() == 1)
			continue;
		istringstream ss(line);
		string id;
		getline(ss, id, ',');
		int classifier_ID = atoi(id.c_str());
		if (classifier_ID > max_classifier_ID)
			max_classifier_ID = classifier_ID;
		classifier_IDs.push_back(classifier_ID);
		vector< vector<float> > confidences_for_behaviors;
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			vector<float> confidences_for_modalities;
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				string conf_str;
				getline(ss, conf_str, ',');
				confidences_for_modalities.push_back( atof(conf_str.c_str()) );
				cout << "......storing confidence " << classifier_ID << " b=" << b_idx << ",m=" << m_idx << ": " << atof(conf_str.c_str()) << "\n";  // debug
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
				ostringstream fn;
				fn << class_fn << "classifier" << classifier_IDs[idx]
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				struct stat buffer;
				if (num_features[b_idx][m_idx] == 0
					|| stat (fn.str().c_str(), &buffer) != 0)
				{
					m_c.push_back(NULL);
					continue;
				}
				cout << "...loaded in " << fn.str().c_str() << "\n";
				CvSVM* c = new CvSVM;
				c->load(fn.str().c_str());
				m_c.push_back(c);
			}
			sub_c.push_back(m_c);
		}
		classifiers[classifier_IDs[idx]] = sub_c;
	}

	// debug
	cout << "... loaded " << classifier_IDs.size() << " classifiers from file\n";

	// debug
	cout << "confidences:\n";
	for (int idx=0; idx < classifier_IDs.size(); idx++)
	{
		cout << classifier_IDs[idx] << ":";
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			cout << b_idx << ":[";
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				cout << confidences[classifier_IDs[idx]][b_idx][m_idx] << ",";
			}
			cout << "];";
		}
		cout << "\n";
	}
	// end debug

	res.success = true;
	return true;
}

// write classifiers and classifier confidences out to files which can later
// be loaded with loadClassifiers
bool saveClassifiers(std_srvs::Empty::Request &,
					 std_srvs::Empty::Response &)
{
	// debug
	cout << "saveClassifiers called\n";

	ofstream conf_file;
	conf_file.open(conf_fn.c_str());

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

				if (num_features[b_idx][m_idx] == 0
					|| confidences[iter->first][b_idx][m_idx] == 0
					|| classifiers[iter->first][b_idx][m_idx] == NULL)
					continue;

				// write classifier out to file
				ostringstream fn;
				fn << class_fn << "classifier" << iter->first
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				classifiers[iter->first][b_idx][m_idx]->save(fn.str().c_str());
			}
		}
		conf_file << '\n';
	}

	conf_file.close();

	// debug
	cout << "... saved classifiers and confidences to file\n";

	return true;
}

// delete classifier and confidence files from disk
bool deleteClassifiers(std_srvs::Empty::Request &,
					   std_srvs::Empty::Response &)
{
	// debug
	cout << "deleteClassifiers called\n";

	for (map<int, vector< vector<CvSVM*> > >::iterator iter = classifiers.begin();
		 iter != classifiers.end(); ++iter)
	{
		//delete all subclassifier files
		for (int b_idx=0; b_idx < num_behaviors; b_idx++)
		{
			for (int m_idx=0; m_idx < num_modalities; m_idx++)
			{
				if (num_features[b_idx][m_idx] == 0 ||
					classifiers[iter->first][b_idx][m_idx] == NULL)
					continue;
				ostringstream fn;
				fn << class_fn << "classifier" << iter->first
					<< "behavior" << b_idx << "modality" << m_idx << ".svm";
				remove(fn.str().c_str());
			}
		}

		// delete classifier confidences file
		remove(conf_fn.c_str());
	}

	// delete any existing classifier pointers
	freeClassifierMemory();
	classifier_IDs.clear();
	max_classifier_ID = 0;

	// debug
	cout << "... deleted classifier and confidence files, freed memory and cleared IDs\n";

	return true;
}

// run a specified classifier on a vector of objects and report results and confidences
bool runClassifier(perception_classifiers::runClassifier::Request &req,
				     perception_classifiers::runClassifier::Response &res)
{
	// debug
	// cout << "classifier " << req.classifier_ID << " for object " << req.object_ID << " called\n";

	// if in cache, just set response params and return
	if (run_classifier_cache.count(req.classifier_ID) == 1 &&
		run_classifier_cache[req.classifier_ID].count(req.object_ID) == 1 &&
		run_classifier_cache[req.classifier_ID][req.object_ID] != NULL)
	{
		res.result = run_classifier_cache[req.classifier_ID][req.object_ID]->result;
		res.confidence = run_classifier_cache[req.classifier_ID][req.object_ID]->confidence;
		res.sub_classifier_decisions = *(run_classifier_cache[req.classifier_ID][req.object_ID]->sub_classifier_decisions);
		return true;
	}

	// run classifier in each relevant behavior, modality combination 
	float decision = 0;
	int sub_classifiers_used = 0;
	vector<float>* _dec = new vector<float>();
	for (int b_idx=0; b_idx < num_behaviors; b_idx++)
	{
		for (int m_idx=0; m_idx < num_modalities; m_idx++)
		{
			if (num_features[b_idx][m_idx] == 0 || classifiers.count(req.classifier_ID) == 0
				|| confidences[req.classifier_ID][b_idx][m_idx] == 0
				|| classifiers[req.classifier_ID][b_idx][m_idx] == NULL)
			{
				_dec->push_back(0);
				continue;
			}

			sub_classifiers_used += 1;
			float _decision = 0;
			float num_positive = 0;
			int observation_count = 1;

			// access feature-getting service and use it to populate rows of test matrix
			Mat test_data;
			perception_classifiers::FetchFeatures ff;
			ff.request.object = req.object_ID;
			ff.request.behavior = b_idx;
			ff.request.modality = m_idx;
			ff.request.allow_missing = false;
			fetch_features.call(ff);
			observation_count = ff.response.rows.size();
			for (int obs_idx=0; obs_idx < ff.response.rows.size(); obs_idx++)
			{
				Mat observation;
				for (int f=0; f < num_features[b_idx][m_idx]; f++)
					observation.push_back(ff.response.rows[obs_idx].features[f]);
				transpose(observation, observation);
				test_data.push_back(observation);
			}

			// run classifier on each observation
			for (int obs_idx=0; obs_idx < ff.response.rows.size(); obs_idx++)
			{
				int response = classifiers[req.classifier_ID][b_idx][m_idx]->predict(test_data.row(obs_idx));
				if (response == 1)
					num_positive += 1.0;
			}

			// average observation decisions to decide this sub classifier's decision
			// could instead do majority voting
			_decision = 2*(num_positive / observation_count) - 1;
			_dec->push_back(_decision);

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
	res.sub_classifier_decisions = *_dec;

	// add to cache
	cache_response* res_cache = new cache_response();
	res_cache->result = res.result;
	res_cache->confidence = res.confidence;
	res_cache->sub_classifier_decisions = _dec;
	run_classifier_cache[req.classifier_ID][req.object_ID] = res_cache;

	// debug
	// cout << "classifier " << req.classifier_ID << " for object " << req.object_ID << ": " << res.result << ", " << res.confidence << "\n";

	return true;
}

// train a classifier with given object IDs and labels and store it under the given classifier ID
bool trainClassifier(perception_classifiers::trainClassifier::Request &req,
				     perception_classifiers::trainClassifier::Response &res)
{
	// debug
	cout << "trainClassifier called for classifier_ID=" << req.classifier_ID << "\n";

	int classifier_ID = req.classifier_ID;
	int num_objects = static_cast<int>(req.object_IDs.size());
	if (std::find(classifier_IDs.begin(), classifier_IDs.end(), classifier_ID) == classifier_IDs.end())
	{
		classifier_IDs.push_back(classifier_ID);
		if (classifier_ID > max_classifier_ID)
			max_classifier_ID = classifier_ID;
	}

	// clear old cache, if any
	if (run_classifier_cache.count(classifier_ID) == 1)
	{
		freeClassifierCache(classifier_ID);
	}

	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// for each behavior and modality, retrieve relevant features for each object and train sub-classifiers
	vector< vector<CvSVM*> > sub_classifiers;
	vector< vector<float> > sub_confidence;
	for (int b_idx=0; b_idx < num_behaviors; b_idx++)
	{
		cout << "...behavior " << b_idx << "\n"; // debug
		vector<CvSVM*> modality_classifiers;
		vector<float> modality_confidence;
		for (int m_idx=0; m_idx < num_modalities; m_idx++)
		{
			cout << "......modality " << m_idx << "\n"; // debug
			//if there are no features for this combination, don't create classifier
			if (num_features[b_idx][m_idx] == 0)
			{
				cout << "......no features\n"; // debug
				modality_classifiers.push_back(NULL);
				modality_confidence.push_back(0);
				continue;
			}

			// retrieve relevant features for each object
			vector<int> num_observations;
			Mat train_data;
			Mat responses;
			bool seen_class_true = false;
			bool seen_class_false = false;
			for (int o_idx=0; o_idx < num_objects; o_idx++)
			{
				// access feature-getting service and use it to populate a row of train_data matrix
				perception_classifiers::FetchFeatures ff;
				ff.request.object = req.object_IDs[o_idx];
				ff.request.behavior = b_idx;
				ff.request.modality = m_idx;
				ff.request.allow_missing = false;
				fetch_features.call(ff);
				num_observations.push_back(ff.response.rows.size());
				for (int obs_idx=0; obs_idx < ff.response.rows.size(); obs_idx++)
				{
					Mat observation;
					for (int f=0; f < num_features[b_idx][m_idx]; f++)
						observation.push_back(ff.response.rows[obs_idx].features[f]);
					transpose(observation, observation);
					train_data.push_back(observation);
					if (req.positive_example[o_idx] == 1)
					{
						responses.push_back(1);
						seen_class_true = true;
					}
					else
					{
						responses.push_back(-1);
						seen_class_false = true;
					}
				}
			}

			float x_fold_correct = 0;
			int cm[2][2] = {{0, 0}, {0, 0}};
			if (seen_class_true && seen_class_false)
			{
				// do leave-one-out cross validation to determine confidence in this classifier
				cout << "......performing cross fold validation\n";  // debug
				CvSVM c_fold;
				for (int fo_idx=0; fo_idx < num_objects; fo_idx++)
				{
					Mat train_fold;
					Mat responses_fold;
					int read_from_row = 0;
					int fold_rows_begin = -1;
					bool _seen_class_true = false;
					bool _seen_class_false = false;
					for (int to_idx=0; to_idx < num_objects; to_idx++)
					{
						if (fo_idx != to_idx)
						{
							for (int obs_idx=0; obs_idx < num_observations[to_idx]; obs_idx++)
							{
								train_fold.push_back(train_data.row(read_from_row+obs_idx));
								responses_fold.push_back(responses.row(read_from_row+obs_idx));
								int v = responses_fold.at<int>(
									responses_fold.size().height-1,responses_fold.size().width-1);
								if (v == 1)
									_seen_class_true = true;
								else
									_seen_class_false = true;
							}
						}
						else
							fold_rows_begin = read_from_row;
						read_from_row += num_observations[to_idx];
					}

					float observations_correct = 0;
					if (_seen_class_true && _seen_class_false)
					{
						c_fold.train(train_fold, responses_fold, Mat(), Mat(), params);
						
						for (int obs_idx=0; obs_idx < num_observations[fo_idx]; obs_idx++)
						{
							int response = c_fold.predict(train_data.row(fold_rows_begin+obs_idx));
							int gsr = responses.at<int>(fo_idx, 0);
							if (response == gsr)
								observations_correct += 1.0;
							int gsrb = gsr;
							if (gsr == -1)
								gsrb = 0;
							int response_b = response;
							if (response == -1)
								response_b = 0;
							cm[gsrb][response_b] += 1;
						}
					}
					float local_correctness = observations_correct / num_observations[fo_idx];
					x_fold_correct += local_correctness;
				}

				// train classifier with all gathered data and store it
				CvSVM* c = new CvSVM;
				cout << "......training primary classifier\n";  // debug
				c->train(train_data, responses, Mat(), Mat(), params);
				modality_classifiers.push_back(c);

			}
			else
			{
				// store a null pointer to the classifier
				cout << "......primary classifier cannot be trained on uniform class data\n";  // debug
				modality_classifiers.push_back(NULL);
			}

			// calculate confidence and store it
			float k = kappa(cm);
			float k_conf = k;
			if (k_conf < 0)
				k_conf = 0.0;
			float confidence;
			if (num_objects > 0)
				confidence = x_fold_correct / static_cast<float>(num_objects);
			else
				confidence = 0;
			//cout << "......primary classifier confidence " << confidence << "\n";  // debug
			// modality_confidence.push_back(confidence);  // use accuracy for confidence
			cout << "......primary classifier confidence " << k_conf << "\n";  // debug
			modality_confidence.push_back(k_conf);  // use kappa statistic for confidence
			
		}

		sub_classifiers.push_back(modality_classifiers);
		sub_confidence.push_back(modality_confidence);
	}
	classifiers[classifier_ID] = sub_classifiers;
	confidences[classifier_ID] = sub_confidence;

	res.success = true;
	return true;
}
