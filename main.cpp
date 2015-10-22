//main.cpp

#include "main.h"


#define REG
//#define FAST

int main(int argc, char* argv[]) {
  int i;
  srand(time(NULL));
  

  // get command line args
  args_t myargs;
  init_args(myargs);
  if (!get_args(argc, argv, myargs)) {
    printf("RT-Rank Version 1.5 (alpha) usage: [-options] train.txt test.txt output.txt [test.txt output.txt]*\n");
	printf("\nRequired flags:\n");
	printf("-f int\tNumber of features in the data sets. \n");
	printf("You must use one of the following:\n");

	printf("\nOptional flags:\n");
	printf("-R \trun ranking (default classification).\n");
	printf("-c \tcost vector input file name, one feature cost per row.\n");
	printf("-m int\tNumber of classes. \n");
//	printf("-M \tcontains missing data. \n"); //current version doesnot support missing data
	printf("-o \tset to 0 if oob samples are NOT used (default); set to 1 if oob samples are used ; set to 2 if oob samples AND validation samples are used for determining node label for test points. \n");
	printf("-a float \t alpha, threashold for impurity function.\n");
	printf("-d int \tmax treep depth.\n");
	printf("-i int \timpurity function: threashold-Pairs(0,default), entropy(1), powers(2).\n");
	printf("-t int \tnumber of trees for random forest.\n");
	printf("-p int \tnumber of processors/threads to use.\n");
	printf("-k int \tnumber of randomly selected features used for building each trees of a random forest.\n");
	printf("-r \tnumber of rounds (/iterations).\n");
	printf("-v \tverbose.\n");
	printf("\n\n");
    return 1;
  }

	// load data from input files
	data_t train;
	vec_data_t test;
	if (!load_data(train, test, myargs)) {
		printf("could not load data files\n"); 
		return 1;
	}
  
	add_idx(train);  
	myargs.ntra=train.size();

	vector<vector<int>> pred1; //based on tree votes from training data
	vector<vector<int>> pred2; //based on tree votes from validation and oob data
	vector<vector<int>> pred3; //based on leaf votes from: (a) training data if oob is not used and not pruning, or (b) validation and oob data if oob is used and pruning
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

	myForest.buildLearn(train, test, myargs);

	myForest.classify(test,myargs, pred1,pred2,pred3, pred1Rank,pred2Rank , pred3Rank, errPrec1, errPrec2, errPrec3, testCostE);

	//write trees to file
	char TreeOutFileName[100];
	sprintf(TreeOutFileName, "%s_%d_trees",myargs.train_file,nTree);
	myForest.writeTrees(TreeOutFileName);

	for(i=0;i<test.size();i++){
		if(myargs.alg==ALG_RANK)
			avgprec2(test[i], pred1Rank[i],pred2Rank[i],pred3Rank[i], 5, errPrec1[i], errPrec2[i],errPrec3[i]);
		cout<< " | test " << i << " | errPrec1: " <<errPrec1[i] << " | errPrec2: " <<errPrec2[i]<<" | errPrec3: "<<errPrec3[i]<< " | CostE: "<< testCostE[i]<<endl;
		char writeResultFileName[100];
		sprintf(writeResultFileName, "%s_%d",myargs.test_outs[i],myargs.trees);

		ofstream out(writeResultFileName);
		out << pred1[i].size() << " " << errPrec1[i] << " " << errPrec2[i] << " " << errPrec3[i] << " "<< testCostE[i] <<endl;
		//turn on the prediction only if necessary to reduce file size
		//	for (int i=0;i<pred1.size();i++)
		//		out << pred1[i] << " " << pred2[i] << " " << pred3[i] << " "<< pred1Rank[i] << " " <<  pred2Rank[i] << pred3Rank[i] << " " <<endl;
		out.close();
	}

  // free memory and exit
	free_memory(myargs, train, test);
return 0;
}
