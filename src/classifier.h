#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:
	vector<string> possible_labels = {"left","keep","right"};
	vector<string> feature_names = {"s", "d", "s_dot", "d_dot"};
	double lane_width = 4.0;

	vector<vector<pair<double, double>>> mean_std_table = {
		{ {0, 0}, {0, 0}, {0, 0} },
		{ {0, 0}, {0, 0}, {0, 0} },
		{ {0, 0}, {0, 0}, {0, 0} },
		{ {0, 0}, {0, 0}, {0, 0} }
	};
	vector<double> priors = { 0, 0, 0};

 	GNB();
 	virtual ~GNB();
 	void train(const vector<vector<double> >& data, const vector<string>&  labels);
  	string predict(const vector<double>&);
};

#endif




