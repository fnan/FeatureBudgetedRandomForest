//main.cpp

#include "main.h"
using namespace std;

#define REG
//#define FAST

int main(int argc, char* argv[]) {
	int i,r;
	srand(time(NULL));

	// get command line args
	args_t myargs;
	init_args(myargs);
	if (!get_args(argc, argv, myargs)) {
		printf("BudgetedRandomForest usage: [-options] train.txt test.txt output.txt [test.txt output.txt]*\n");
		printf("\nRequired flags:\n");
		printf("-f int\tNumber of features in the data sets. \n");
		printf("You must use one of the following:\n");
		printf("\nOptional flags:\n");
		printf("-R \trun ranking (default classification).\n");
		printf("-c \tcost vector input file name, one feature cost per row.\n");
		printf("-m int\tNumber of classes. \n");
		//	printf("-M \tcontains missing data. \n");
		printf("-o \t0(default) if in-bag samples only are used for estimating node error; set to 1 if oob samples only are used for estimating node error; \n set to 2 if both in-bag and oob samples are used for estimating node error. \n");
		printf("-a float \t alpha, threashold for impurity function.\n");
		printf("-u float \t pruning trade-off parameter. Pruning turned on if non-negative (requires tree_input_file) \n");
		printf("-d int \tmax treep depth.\n");
		printf("-i int \timpurity function: threashold-Pairs(0,default), entropy(1), powers(2).\n");
		printf("-t int \tnumber of trees for random forest.\n");
		printf("-p int \tnumber of processors/threads to use.\n");
		printf("-k int \tnumber of randomly selected features used for building each trees of a random forest.\n");
		printf("-v \tverbose.\n");
		printf("-b \t learning rate base: larger value leads to larger step size. default: 500.\n");
		printf("-S \t Bagging ratio, (0,1]\n");
		printf("-g \t costgroup file, optional. \n");
		printf("\n\n");
		return 1;
	}
	double avgPred1=0.0, avgCostE=0.0, squaredErr=0.0,squaredCost=0.0;


	for(r=0;r<myargs.rounds;r++){ //perform rounds-fold cross validation, default rounds=1: no cross validation
		data_t train;
		vec_data_t test;
		if (!load_data(train, test, myargs,r)) {// load data from input files
			printf("could not load data files\n"); 
			return 1;
		}
  
		add_idx(train);  
		myargs.ntra=train.size();

		vector<vector<int>> pred1; //based on tree votes from training data
		vector<vector<int>> pred2; //based on tree votes from validation and oob data
		vector<vector<int>> pred3; //based on leaf votes from training data
		vector<vector<double>> pred1Rank; //based on tree votes from training data
		vector<vector<double>> pred2Rank; //based on tree votes from validation and oob data
		vector<vector<double>> pred3Rank; //based on leaf votes from: (a) training data if oob is not used and not pruning, or (b) validation and oob data if oob is used and pruning
		vector<double> errPrec1(test.size(),0.0);
		vector<double> errPrec2(test.size(),0.0);
		vector<double> errPrec3(test.size(),0.0);
		vector<double> testCostE(test.size(),0.0);

		int num_c=myargs.num_c;
		int nTree=myargs.trees;
		BudgetForest myForest(nTree,num_c);

		/*perform feature analysis only, no training nor pruning. tree_file name is provided in input*/

		if(false && myargs.analysis){	
			if(myForest.readTreesFromFile(myargs.tree_file)){
				std::cout<< "Error reading tree file: "<< myargs.tree_file<<endl;
				return 1;
			}

			int verbOld=myargs.verbose;
				myargs.verbose=2;  //ask classify() to write the featureMatrix to file
			myForest.classify(test,myargs, pred1,pred2,pred3, pred1Rank,pred2Rank , pred3Rank, errPrec1, errPrec2, errPrec3, testCostE,r);
		//		myargs.verbose=verbOld;
		//		myForest.featureTreeCount(myargs,r);
			for(i=0;i<test.size();i++){
				if(myargs.alg==ALG_RANK)
					avgprec2(test[i], pred1Rank[i],pred2Rank[i],pred3Rank[i], 5, errPrec1[i], errPrec2[i],errPrec3[i]);
				std::cout<< " | test " << i << " | errPrec1: " <<errPrec1[i] << " | errPrec2: " <<errPrec2[i]<<" | errPrec3: "<<errPrec3[i]<< " | CostE: "<< testCostE[i]<<endl;
			}
			/* skip to the end */
			free_memory(myargs, train, test);
			continue;
		}

		//write trees to file
		char TreeOutFileName[100];
		if(myargs.prune>=0.0)
			sprintf(TreeOutFileName, "%s_%d_%d_%d_%f_PruTr_%d",myargs.train_file,r,nTree,myargs.loss,(myargs.prune),myargs.pruneMethod);
		else{
			sprintf(TreeOutFileName, "%s_%d_%d_%d_trees",myargs.train_file,r,nTree,myargs.loss);
			if(myargs.alg==ALG_BOOST_EXPSPLIT)
				sprintf(TreeOutFileName, "%s_%d_%d_%d_trees_d%dEXP",myargs.train_file,r,nTree,myargs.loss, myargs.depth);
			if(myargs.alg==ALG_BOOST_MAXSPLIT)
				sprintf(TreeOutFileName, "%s_%d_%d_%d_trees_d%dMAX",myargs.train_file,r,nTree,myargs.loss, myargs.depth);			
		}
		if(myargs.prune>=0.0){//pruning mode: requires tree_input_file
			char readTreeFileName[100];
			if(myargs.pruneMethod<0) //prune based on previously pruned tree
				sprintf(readTreeFileName, "yahoo_cs_tr_140_0.400000_PruTrShort_2");
							//	sprintf(readTreeFileName, "%s_%d_CCPruTr",myargs.train_file,nTree);
			else
				sprintf(readTreeFileName, "%s_%d_%d_%d_trees",myargs.train_file,r,nTree,myargs.loss);
			char writeLPFileName[100];

			/*prune ensemble with LP*/

			if(myargs.pruneMethod==0){
				if(myargs.incre==0){ //pruning the entire forest
					if(myForest.readTreesFromFile(readTreeFileName)){
						std::cout<< "Error reading tree file: "<< readTreeFileName<<endl;
						return 1;
					}
					//assuming test[0] is validation set
					sprintf(writeLPFileName, "%s_%d_%d_LP2.mps",myargs.train_file,r,nTree);
					if(myargs.costgroup_file==NULL){ //no group
						myForest.pruneGA(myargs,test[0],writeLPFileName);
					}
					else
						myForest.pruneGAgrp(myargs, test[0],writeLPFileName);
				}
			}
		}
		else{//training mode: build decision trees
			if(myargs.analysis){	
				if(myForest.readTreesFromFile(myargs.tree_file)){
					std::cout<< "Error reading tree file: "<< myargs.tree_file<<endl;
					return 1;
				}
			}
			else
				myForest.buildLearn(train, myargs);
		}
	
		myForest.classify(test,myargs, pred1,pred2,pred3, pred1Rank,pred2Rank , pred3Rank, errPrec1, errPrec2, errPrec3, testCostE,r);

		if(myargs.prune<0.0 || myargs.verbose>0)
			myForest.writeTrees(TreeOutFileName,0);

		for(i=0;i<test.size();i++){
			if(myargs.alg==ALG_RANK)
				avgprec2(test[i], pred1Rank[i],pred2Rank[i],pred3Rank[i], 5, errPrec1[i], errPrec2[i],errPrec3[i]);
			std::cout<< " | test " << i << " | errPrec1: " <<errPrec1[i] << " | errPrec2: " <<errPrec2[i]<<" | errPrec3: "<<errPrec3[i]<< " | CostE: "<< testCostE[i]<<endl;
		}
		squaredErr+=errPrec1[1]*errPrec1[1];
		squaredCost+=testCostE[1]*testCostE[1];
		avgPred1+=errPrec1[1];
		avgCostE+=testCostE[1];
		/*compute the compression ratio of the pruning*/
		int numInternalNode=myForest.numInternalNodes();
		int totalNodes=0;
		for(i=0;i<myForest.nTrees;i++){
			totalNodes+=myForest.nNodes[i];
		}
		char writeFinalResultFileNameVal[100];
		if(myargs.prune>=0.0)
			sprintf(writeFinalResultFileNameVal, "%s_Val_%d_%d_%f_%f_%d_OUT",myargs.train_file,myargs.trees,myargs.loss,myargs.prune,myargs.alpha,myargs.pruneMethod);
		else
			sprintf(writeFinalResultFileNameVal, "%s_Val_%d_%d_%f_%f_-1_OUT",myargs.train_file,myargs.trees,myargs.loss,myargs.prune,myargs.alpha);
		int val_id = test.size() - 2;
		ofstream ofs;
		ofs.open(writeFinalResultFileNameVal, std::ofstream::app);
		ofs << errPrec1[val_id] << " " << errPrec2[val_id] << " " << errPrec3[val_id] << " " << testCostE[val_id] << " " << (double)(numInternalNode * 2 + nTree) / totalNodes << " " << (int)myargs.loss << " " << myargs.alpha << endl;
		ofs.close();

		int test_id = test.size() - 1;
		char writeFinalResultFileNameTe[100];
		if(myargs.prune>=0.0)
			sprintf(writeFinalResultFileNameTe, "%s_Te_%d_%d_%f_%f_%d_OUT",myargs.train_file,myargs.trees,myargs.loss,myargs.prune,myargs.alpha,myargs.pruneMethod);
		else
			sprintf(writeFinalResultFileNameTe, "%s_Te_%d_%d_%f_%f_-1_OUT",myargs.train_file,myargs.trees,myargs.loss,myargs.prune,myargs.alpha);

		ofs.open(writeFinalResultFileNameTe, std::ofstream::app);
		ofs << errPrec1[test_id] << " " << errPrec2[test_id] << " " << errPrec3[test_id] << " " << testCostE[test_id] << " " << (double)(numInternalNode * 2 + nTree) / totalNodes << " " << (int)myargs.loss << " " << myargs.alpha << endl;
		ofs.close();

		/*output the pruned trees*/
		if(myargs.pruneMethod>=0){
			sprintf(TreeOutFileName, "%s_%d_%d_%f_PruTrShort_%d",myargs.train_file,nTree,myargs.loss,myargs.prune,myargs.pruneMethod);
			myForest.writeCCP(TreeOutFileName);
		}
		free_memory(myargs, train, test);
	}

  avgPred1=avgPred1/r;
  avgCostE=avgCostE/r;
  squaredErr=squaredErr/r;
  squaredCost=squaredCost/r;

	std::cout<< "AvgCost: " << avgCostE<< ", StdDevCost: "<< sqrt(squaredCost-avgCostE*avgCostE)<<", AvgErr: " << avgPred1  <<", StdDevErr: "<< sqrt(squaredErr-avgPred1*avgPred1) << endl;
return 0;
}
