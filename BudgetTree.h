#ifndef BUDGETTREE_H
#define BUDGETTREE_H

#include <unordered_map>
#include <random>
#include <cmath>
#include <ctime>
#include "lp_lib.h"
#include "BudgetNode.h"

using namespace std;

struct pruningInfo{
	vector<vector<unordered_map<int,int>>>* featUsageSubTreeSC;
	vector<vector<unordered_map<int,int>>>* featUsageSubTreeDC;
	vector<vector<int>>* numLeavesSubTree;
	vector<vector<int>>* errNode;
	vector<vector<int>>* errSubTree;
	vector<vector<vector<int>>>* examplesInNode;
	vector<vector<int>>* exampleFeatCount;
};

struct threadingRange{
	data_t dataInp;
	int start;
	int end;
	int nTreesGating;
};

class BudgetTree{
private:
	int numberOfClasses;


public:
	int nNodes;
	vector<BudgetNode*> nodePts;
	double prunedAway;
	vector<int> leafIndex; // leaf ID for each node - length == nNodes
	vector<int> leafIndexReverse; // length == nLeaf
	vector<int> leafPerExample; // leaf ID for each training example - used for re-training the weights in gradient boosting
	int nLeaf; //number of leaves

	BudgetTree(int nNodes=1, int numberOfClasses=2);
	~BudgetTree();
	int buildTreeFromFile(ifstream& input,BudgetNode*& parNode);
	void prune(double lambda, const data_t& val, const vector<double>& costV, char* writeLPFileName, int verbose, int oob);
	void BudgetTree::CCPrune(double compressRatio,int oob);
	void copyOob();
	void writeTree(const BudgetNode* curNode, ofstream& outfile);
	void buildLearn(const data_t& train, args_t& args, double (*impurityHandle)(int,vector<int>&,double),std::default_random_engine& gen);
	void oobUpdate(const data_t& train, const data_t& sample);
	void ValUpdate(const data_t& val);
	void reassignID(BudgetNode* curNode, int* curID, int parentID);
	void writePrunedTree(const BudgetNode* curNode, ofstream& outfile);
	void BudgetTree::predSimple(const data_t& train, const args_t& args, vector<double>& pred);
	void BudgetTree::predInRangeSimple(const data_t& data, threadingRange& rangeInd, vector<double>& pred);
	void updateNodePts();
	void traverseTree(BudgetNode* curNode);
	void BudgetTree::updateLeafindex(const data_t& data);
	void BudgetTree::getLeafPred(vector<double>& predTmp);
};
#endif