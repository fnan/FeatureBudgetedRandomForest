#ifndef BUDGETFOREST_H
#define BUDGETFOREST_H

#include "BudgetTree.h"
#include <cmath>
#include <ctime>
#include <string>

using namespace boost;

class BudgetForest{
private:
	int numberOfClasses;

	struct PredOutputs {
		vector<int> pred1_tmp;
		vector<int> pred2_tmp;
		vector<int> pred3_tmp;
		vector<double> pred1Rank_tmp;
		vector<double> pred2Rank_tmp;
		vector<double> pred3Rank_tmp;
	};
	void classifyInRange(data_t test,  const args_t& args, int test_start, int test_end, struct PredOutputs& outStruct, bool** testFeatureUsed);

public:
	int nTrees;
	vector<BudgetTree*> treePts;
	vector<int> nNodes;	

	BudgetForest(int nTrees=1, int numberOfClasses=2);
	~BudgetForest();
	int readTreesFromFile(char* treesFileName);
	void copyOob();
	void classify(const vec_data_t& test, args_t& args, vector<vector<int>>& pred1,vector<vector<int>>& pred2,vector<vector<int>>& pred3,vector<vector<double>>& pred1Rank,vector<vector<double>>& pred2Rank ,vector<vector<double>>& pred3Rank , vector<double>& errPrec1, vector<double>& errPrec2,vector<double>& errPrec3,vector<double>& costE);
	void writeTrees(char* treesFileName);
	void buildLearn(const data_t& train, const vec_data_t& test, args_t& args);
	void buildPerProc(int nTreesPerProc, const data_t& train, const vec_data_t& test, args_t& args, vector<BudgetTree*>& treesPerProc);
};
#endif