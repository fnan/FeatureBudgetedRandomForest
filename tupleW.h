#ifndef AM_ML_TUPLE_H
#define AM_ML_TUPLE_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <map>

using namespace std;

#define MY_DBL_MAX 99999999999999.999999
#define MY_DBL_MIN (-MY_DBL_MAX)
#define UNKNOWN MY_DBL_MIN


class tupleW // represents a data instance
{
public:
	double* features;
	int label;
	int psLabel;
	double weight;
	int qid;
	double pred;
	int idx;

tupleW(int num_features) : weight(1.0), label(-1), qid(-1), pred(-1), psLabel(0){
	features = new double[num_features+1];
	for (int i = 0; i <= num_features; i++) 
		features[i] = UNKNOWN;
}
  ~tupleW() {delete[] features;}

//  static int read_input(vector<tupleW*>& data, char* file, int num_features, bool training,  bool ranking); 
// populate vector with tuples from input file
// input: file is of format "label qid:int feature1:value1 f2:v2 f5v5 ...."
// output: creation of tuples (one for each line in input file), into data vector
 static int read_input(vector<tupleW*>& data, char* file, int num_features, bool training, bool ranking) 
{
	ifstream input(file);
	if (input.fail())
		return 0;

	//  static double* init_values = init_values = tupleW::read_default_features(missing_file, num_features);
 
	string strline;
	int idx=0;
	while( getline(input, strline) ) {
		tupleW* t = new tupleW(num_features);
		if (t == NULL)
		  cout << "out of memory" << endl;

		char* line = _strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
		char* tok = NULL;

		if (training) { // if this is a training set, extract label (first item)
		  tok = strtok(line, " ");
		  t->label = atoi(tok);
		  t->pred = (t->label ==0) ? -1.0 : 1.0; 
		  t->weight = 1;
		}

		int first = 1;
		while (tok = strtok(training ? NULL : tok ? NULL : line, " \n")) { // tok is feature:value
		  string bit = tok;
		  int colon_index = bit.find(":");
		  string feature = bit.substr(0, colon_index);
		  string value = bit.substr(colon_index+1, bit.length()-colon_index-1);
		  if (first&& ranking) { t->qid = atoi(value.c_str()); first = 0; continue; } // first token is query
		  int f = atoi(feature.c_str());
		  double v = (double)atof(value.c_str());
		  t->features[f] = v;
		}
		t->features[0]=idx++;
		free(line);
		data.push_back(t);
  }  
  return 1;
}

  // free memory
  static void delete_data(vector<tupleW*>& data) {
    for (int i = 0; i < data.size(); i++) {
		delete data[i];
	}
  }
};

#endif //AM_ML_TUPLE_H
