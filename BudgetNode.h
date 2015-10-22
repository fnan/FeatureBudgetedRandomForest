#ifndef BUDGETNODE_H
#define BUDGETNODE_H

#include "tupleW.h"
#include "args.h"
#include "impurity.h"
#include <map>
#include <boost\numeric\ublas\vector.hpp>
#include <boost\bind.hpp>
#include <boost\thread\thread.hpp>

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
	int pred; // what this node predicts if leaf node
	boost::numeric::ublas::vector<int> c_leaf; //number of examples in each class if it is a leaf
	boost::numeric::ublas::vector<int> oob_c_leaf; //number of examples in each class using oob samples

	BudgetNode(int num_c){
		int i;
		for (i = 0; i < CHILDTYPES; i++)
			child[i] = 0;
		c_leaf.resize(num_c,false);
		oob_c_leaf.resize(num_c,false); 
	}
	BudgetNode(const vector<tupleW*>& data, args_t& myargs, int depth, int fk, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double), int* curID, int parentID) : leaf(false), feature(0), value(UNKNOWN), pred(-1){    
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
		if(myargs.loss==ALG_ENTROPY){
			if (!impurity_splitE_noMiss(data, f_split, v_split, imp, c_leaf, myargs, impurityHandle)){//fk: number of randomly selected features used for building each trees of a random forest			
				leaf = true;
				return;   	
			}
		}
		else{
			if (!impurity_splitW_noMiss(data, f_split, v_split, imp, c_leaf, myargs, impurityHandle)){//fk: number of randomly selected features used for building each trees of a random forest			
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
		child[YES] = new BudgetNode(child_data[YES], myargs, depth+1, fk, impurityHandle, curID, ID);
		child[NO] = new BudgetNode(child_data[NO], myargs, depth+1, fk, impurityHandle, curID, ID);

		//if (child_data[MISSING].size())
		//	child[MISSING] = new BudgetNode(child_data[MISSING], myargs, depth+1, fk, impurityHandle);
		//else
		//	child[MISSING] = 0;
	}
 
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
};

#endif //AM_DECISION_TREE_ML_H
