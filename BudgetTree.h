#ifndef BUDGETTREE_H
#define BUDGETTREE_H
#include "BudgetNode.h"
#include <boost\random\mersenne_twister.hpp>
#include <boost\random\uniform_int_distribution.hpp>
#include <random>
using namespace std;

class BudgetTree{
private:
	int numberOfClasses;

	void updateNodePts();
	void traverseTree(BudgetNode* curNode);

public:
	int nNodes;
	vector<BudgetNode*> nodePts;

	BudgetTree(int nNodes=1, int numberOfClasses=2);
	~BudgetTree();
	int buildTreeFromFile(ifstream& input,BudgetNode*& parNode);
	void setNNodes(int nNodes);
	void copyOob();
	void writeTree(const BudgetNode* curNode, ofstream& outfile);
	void buildLearn(const data_t& train, const vec_data_t& test,args_t& args, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double),std::default_random_engine& gen);
	void oobUpdate(const data_t& train, const data_t& sample);
	void ValUpdate(const data_t& val);
};
#endif