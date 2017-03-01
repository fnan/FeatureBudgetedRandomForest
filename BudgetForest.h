#ifndef BUDGETFOREST_H
#define BUDGETFOREST_H

#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <ilcplex/cplex.h>
#include <ilcplex/cplexx.h>
#include <set>
#include <list>
#include "lp_lib.h"
#include "BudgetTree.h"


struct pruningInp{
	int start;
	int numTrees;
};

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
		vector<vector<int>> leafIndex;
		vector<vector<double>> proba_pred;
	};
	void BudgetForest::classifyInRange(threadingRange threadingInp, const args_t& args, struct PredOutputs& outStruct, int** testFeatureUsed);
	void BudgetForest::updateExampleFeatCount(int treeID, int curNodeID, const vector<vector<vector<int>>>& examplesInNode, vector<vector<int>>& exampleFeatCount);
	void BudgetForest::solveNetworkPerProc(vector<CPXENVptr>& envs, vector<CPXNETptr>& networks, const pruningInp pruneInp, int& status);
	void BudgetForest::avgprec2Simple(const data_t& data, vector<double>& predRank, int topX, double& prec);

public:
	int nTrees;
	vector<BudgetTree*> treePts;
	vector<int> nNodes;	
	vector<vector<int>> leafIndex; //index map from classifier node ID to gating node pointer index
	double wt; //class weight to balance pseudo label class

	BudgetForest(int nTrees=1, int numberOfClasses=2);
	~BudgetForest();
	int readTreesFromFile(char* treesFileName);
	int BudgetForest::readTreesFromFileIncre(char* treesFileName, int start, int incre);
	void copyOob();
	void BudgetForest::pruneGAgrp(const args_t& myargs, const data_t& val,  char* writeLPFileName);
	void BudgetForest::pruneGA(const args_t& myargs, const data_t& val,  char* writeLPFileName);
	void classify(const vec_data_t& test, args_t& args, vector<vector<int>>& pred1,vector<vector<int>>& pred2,vector<vector<int>>& pred3,vector<vector<double>>& pred1Rank,vector<vector<double>>& pred2Rank ,vector<vector<double>>& pred3Rank , vector<double>& errPrec1, vector<double>& errPrec2,vector<double>& errPrec3,vector<double>& costE, int r);
	void writeTrees(char* treesFileName,int append);
	void buildLearn( data_t& train,args_t& args);
	void buildPerProc(int nTreesPerProc, const data_t& train,  args_t& args, vector<BudgetTree*>& treesPerProc);
	void BudgetForest::featureTreeCount(args_t& args,int r);
	int BudgetForest::numInternalNodes();
	void BudgetForest::writeCCP(char* treesFileName);
	void BudgetForest::fillExampleFeatCountInRange(const data_t& train, const args_t& myargs, int test_start, int test_end, vector<vector<int>>& exampleFeatCount);
};
#endif