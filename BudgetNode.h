#ifndef BUDGETNODE_H
#define BUDGETNODE_H

#include <map>
#include <random>
#include "tupleW.h"
#include "args.h"
#include "impurity.h"

using namespace std;

class BudgetNode
{
public:
	enum {YES, NO, MISSING, CHILDTYPES};

	int ID,parID; //self and parent IDs
	BudgetNode* child[CHILDTYPES];
	int feature;
	double value; // feature and value this node splits on
	bool leaf;
	int pred; // what this node predicts if leaf node: int for classification
	double pred_double; // what this node predicts if leaf node: double for Boosting
	vector<int> c_leaf; //number of examples in each class if it is a leaf
	vector<int> oob_c_leaf; //number of examples in each class using oob samples
	double stopP; //stop probability - from LP pruning solution
	int depth;

	BudgetNode(int num_c){
		int i;
		for (i = 0; i < CHILDTYPES; i++)
			child[i] = 0;
		c_leaf.resize(num_c,false);
		oob_c_leaf.resize(num_c,false); 
		stopP=0.0;
	}
	//overload constructor for boosting
	BudgetNode(const vector<tupleW*>& data, const args_t& myargs, int depth,  double (*impurityHandle)(vector<double>&), int* curID, int parentID) : leaf(false), feature(0), value(UNKNOWN), pred(-1), stopP(0.0), pred_double(0.0){
		int i, ndata=data.size();
		int maxdepth=myargs.depth, num_features=myargs.features-1;
		double imp=0.0;
		for (i = 0; i < CHILDTYPES; i++)
			child[i] = 0;
		parID=parentID;
		ID=*curID;
		*curID=ID+1;
		this->depth = depth;
		
		c_leaf.resize(myargs.num_c,false);
		oob_c_leaf.resize(myargs.num_c,false); 

		// copy the data within the current node for parallel feature search
		vector<vector<double>> dataMatrix(myargs.features, vector<double>(ndata,0.0)); //sample weight included as an additional feature field
		vector<double> dataTargets(ndata,0.0);
		for(i =0;i<ndata;i++){
			dataTargets[i]=data[i]->pred;
			for(int j=0;j<num_features;j++)
				dataMatrix[j][i]=data[i]->features[j+1];
			dataMatrix.back()[i]=data[i]->weight;
		}
		
		// get prediction
		pred_double= std::accumulate(dataTargets.begin(), dataTargets.end(),0.0)/ndata;
		pred_double=pred_double*myargs.alpha; //multiplying learning rate

		imp=(*impurityHandle)(dataTargets);
		// check if leaf node 
		// return if measure impurity ==0
		if ( imp<=1e-10 || depth >= maxdepth) {
			leaf = true;
			return;
		}
		int f_split;
		double v_split;
		// split data into 3 parts, based on criteria found
		vector<tupleW*> child_data[CHILDTYPES];
		split_data_noMiss(data, child_data, f_split, v_split, myargs);
		//printf("split: %d %f, Y:%d N:%d  M:%d\n", f_split, (float)v_split, child_data[YES].size(), child_data[NO].size(), child_data[MISSING].size());

		if (!(child_data[YES].size() && child_data[NO].size())) {
			leaf = true;
			return;
		}

		// remember where we splitted, and recurse
		feature = f_split;
		value = v_split;
		child[MISSING] = 0; //assume no missing data
		child[YES] = new BudgetNode(child_data[YES], myargs, depth+1, impurityHandle, curID, ID);
		child[NO] = new BudgetNode(child_data[NO], myargs, depth+1, impurityHandle, curID, ID);
	}
	BudgetNode(const vector<tupleW*>& data, args_t& myargs, int depth,  double (*impurityHandle)(int, vector<int>&,double), int* curID, int parentID,std::default_random_engine& gen) : leaf(false), feature(0), value(UNKNOWN), pred(-1), stopP(0.0){    
		int i, ndata=data.size();
		int num_c=myargs.num_c;
		int maxdepth=myargs.depth;
		double imp=0.0, alpha=myargs.alpha;
		for (i = 0; i < CHILDTYPES; i++)
			child[i] = 0;
		parID=parentID;
		ID=*curID;
		*curID=ID+1;
		
		c_leaf.resize(num_c,false);
		oob_c_leaf.resize(num_c,false); 
		// get prediction
		for (i=0;i<num_c;i++) {
			c_leaf[i]=0;
			oob_c_leaf[i]=0;
		}
		for (i=0;i<ndata;i++)
			c_leaf[data[i]->label]++;
		int maxTmp=0;
		for (i=0;i<num_c;i++){
			if (c_leaf[i]>maxTmp){
				maxTmp=c_leaf[i];
				pred=i;
			}
		}

		imp=(*impurityHandle)(num_c, c_leaf, alpha);
		// check if leaf node 
		// return if measure impurity ==0
		if (imp<=0 || depth >= maxdepth) {
			leaf = true;
			return;
		}
		int f_split; double v_split;
		vector<int> featureSet;
		if(myargs.kfeatures<0){
			for (i=0; i < myargs.features-1; i++)
				featureSet.push_back(i+1);			
		}
		else{
			vector<int> featureTmp;
			for (i=0; i < myargs.features-1; i++)
				featureTmp.push_back(i+1);			
			int maxInd=myargs.features-2; //index pointing to the end of the featureTmp
			int curIndPick, swapTmp;
			for (i=0; i < myargs.kfeatures; i++){
				std::uniform_int_distribution<int> dist(0,maxInd);
				curIndPick=dist(gen);
				swapTmp=featureTmp[maxInd];
				featureTmp[maxInd]=featureTmp[curIndPick];
				featureTmp[curIndPick]=swapTmp;
				maxInd--;
			}
			for(i=maxInd+1;i<myargs.features-1;i++)
				featureSet.push_back(featureTmp[i]);
		}
		if(myargs.loss==ALG_ENTROPY){
			if (!impurity_splitE_noMiss(data, f_split, v_split, imp, c_leaf, myargs, impurityHandle,featureSet)){
				leaf = true;
				return;   	
			}
		}
		else if(myargs.alg==ALG_BOOST_MAXSPLIT || myargs.alg==ALG_BOOST_EXPSPLIT){
			// copy the data within the current node for parallel feature search
			vector<vector<double>> dataMatrix(featureSet.size(), vector<double>(ndata,0.0));
			for(i =0;i<ndata;i++)
				for(int j=0;j<myargs.kfeatures;j++)
					dataMatrix[j][i]=data[i]->features[featureSet[j]];
			
		}
		else{
			if (!impurity_splitW_noMiss(data, f_split, v_split, imp, c_leaf, myargs, impurityHandle,featureSet)){
				leaf = true;
				return;   
			}
		}
		// split data into 3 parts, based on criteria found
		vector<tupleW*> child_data[CHILDTYPES];
		split_data_noMiss(data, child_data, f_split, v_split, myargs);
		//printf("split: %d %f, Y:%d N:%d  M:%d\n", f_split, (float)v_split, child_data[YES].size(), child_data[NO].size(), child_data[MISSING].size());

		if (!(child_data[YES].size() && child_data[NO].size())) {
			leaf = true;
			return;
		}

		// remember where we splitted, and recurse
		feature = f_split;
		value = v_split;
		child[MISSING] = 0; //assume no missing data
		child[YES] = new BudgetNode(child_data[YES], myargs, depth+1, impurityHandle, curID, ID,gen);
		child[NO] = new BudgetNode(child_data[NO], myargs, depth+1, impurityHandle, curID, ID,gen);

		//if (child_data[MISSING].size())
		//	child[MISSING] = new BudgetNode(child_data[MISSING], myargs, depth+1, impurityHandle);
		//else
		//	child[MISSING] = 0;
	}
 
//	static double classify(const tupleW* const instance, const BudgetNode* const dt);
//	static double boosted_classify(const vector< BudgetNode* >&, const tupleW* const, double alpha);
	// input: vector of tuples (data), children vector, feature and value to split on
	// output: split tuples in data into appropriate children vectors
	static void split_data(const vector<tupleW*>& data, vector<tupleW*> child[CHILDTYPES], int f, double v, const args_t& myargs) {
		int n = data.size(), i;
		for (i = 0; i < CHILDTYPES; i++)
			while(child[i].size())
				child[i].pop_back();

		if (myargs.missing==1){ //if there is missing data
			for (i = 0; i < n; i++) 
				if (data[i]->features[f] == UNKNOWN)
					child[MISSING].push_back(data[i]);
				else if (data[i]->features[f] <= v)
					child[YES].push_back(data[i]);
				else
					child[NO].push_back(data[i]);
		}
		else {
			for (i = 0; i < n; i++) 
				if (data[i]->features[f] <= v)
					child[YES].push_back(data[i]);
				else
					child[NO].push_back(data[i]);		
		}
	}

	static void split_data_noMiss(const vector<tupleW*>& data, vector<tupleW*> child[CHILDTYPES], int f, double v, const args_t& myargs) {
		int n = data.size(), i;
		for (i = 0; i < CHILDTYPES; i++)
			while(child[i].size())
				child[i].pop_back();

			for (i = 0; i < n; i++) 
				if (data[i]->features[f] <= v)
					child[YES].push_back(data[i]);
				else
					child[NO].push_back(data[i]);
	}
	static void split_dataIndex_noMiss(const data_t& data, vector<int>& dataInd, vector<int> child[CHILDTYPES], int f, double v, const args_t& myargs) {
		int n = dataInd.size(), i;
		for (i = 0; i < CHILDTYPES; i++)
			while(child[i].size())
				child[i].pop_back();

			for (i = 0; i < n; i++) 
				if (data[dataInd[i]]->features[f] <= v)
					child[YES].push_back(dataInd[i]);
				else
					child[NO].push_back(dataInd[i]);
	}
};

#endif //AM_DECISION_TREE_ML_H
