#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {}

GNB::~GNB() {}

double my_mean(const vector<double>& x) {
	return accumulate(x.begin(), x.end(), 0.0)/x.size();
}

double stddev(const vector<double>& x, double mean) {
	vector<double> xc = x;

	transform(xc.begin(), xc.end(), xc.begin(), [&](double v){ return (v - mean)*(v - mean); });

	return sqrt(my_mean(xc));
}

double gaussian(double x, double mu, double sigma) {
	return 1./sqrt(2.*M_PI*sigma*sigma)*exp(-(x-mu)*(x-mu)/(2.*sigma*sigma));
}

void GNB::train(const vector<vector<double>>& data, const vector<string>& labels)
{
	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
    for(auto label : labels) {
		for(size_t i = 0; i < priors.size(); i++) {
            if (label == possible_labels[i]) {
		        priors[i] += 1.;
			}
		}
	}
	for(size_t i = 0; i < priors.size(); i++)
	    priors[i] /= labels.size();

	for(size_t i = 0; i < priors.size(); i++)
		cout << possible_labels[i] << "  " << priors[i] << endl;

	for(size_t i = 0; i < data[0].size(); i++) {
		for(size_t j = 0; j < possible_labels.size(); j++) {
            vector<double> x;
			for(size_t k = 0; k < data.size(); k++) {
				if (labels[k] == possible_labels[j]) {
                    x.push_back(data[k][i]);
				}
			}
			double m = my_mean(x), s = stddev(x, m);
			mean_std_table[i][j] = {m, s};
		}
	}
    cout << "Feature/label table:" << endl;
	for(size_t i = 0; i < data[0].size(); i++) {
		for(size_t j = 0; j < possible_labels.size(); j++) {
			cout << feature_names[i] << "/" << possible_labels[j] << ": ";
			cout << mean_std_table[i][j].first << "," << mean_std_table[i][j].second << endl;
		}
	}
}

string GNB::predict(const vector<double>& sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	vector<double> label_probabilities = {0,0,0};
    double max_probability = 0;
	size_t l = 0;
	for(size_t k = 0; k < possible_labels.size(); k++) {
		double product = priors[k];
	    for(size_t i = 0; i < sample.size(); i++) {
		    product *= gaussian(sample[i], mean_std_table[i][k].first, mean_std_table[i][k].second);
		}
		label_probabilities[k] = product;
		if (product > max_probability) {
		    max_probability = product;
			l = k;
		}
	}

	return possible_labels[l];
}
