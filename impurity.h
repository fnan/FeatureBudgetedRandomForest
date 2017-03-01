#ifndef FN_IMPURITY_ML_H
#define FN_IMPURITY_ML_H

// ####
//#include "getopt.h"

// ###
#include "args.h"

#include <numeric>
#include <thread>
#include "tupleW.h"


using std::vector;

typedef struct searchSplitParam{
	int start;
	int end;
	alg_t alg;
	double totalWeights;
};

void evalFeatureSplitsPerProc(const vector<vector<double>>& dataMatrix, const vector<double>& dataTargets, vector<double>& impReduction, vector<double>& v_splits,  searchSplitParam& param);
bool impurity_splitW_noMiss(data_t data, int& f_split, double& v_split, double imp, vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, vector<int>&,double),vector<int> featureSet);
bool impurity_splitE_noMiss(data_t data, int& f_split, double& v_split, double imp, vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, vector<int>&,double),vector<int> featureSet);
double impurityEntropy(int num_c, vector<int>& c_tmp, double alpha);
double impurityHP(int num_c, vector<int>& c_tmp, double alpha);
double impurityMeanSq(vector<double>& targets);
double impurityDeviance(vector<double>& targets);
void logLoss(data_t& data, const vector<double>& currentPred, double& loss);
void pseudoLogLoss(data_t& data, const vector<double>& currentPred, double& loss);
vector<int> sort_indexes(const vector<double> &v);
#endif