#include "BudgetForest.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

BudgetForest::BudgetForest(int nTrees, int numberOfClasses){
	this->nTrees=nTrees;
	this->numberOfClasses=numberOfClasses;
	nNodes.resize(nTrees,1);
	treePts.resize(nTrees,NULL);
}
/* construct trees from treesFileName */
int BudgetForest::readTreesFromFile(char* treesFileName){
	int i;
	ifstream input(treesFileName);
	if (input.fail())
		return 1;

	string strline;
	for(i=0;i<nTrees;i++){		
		getline(input, strline);
		char* line = _strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
		char* tok = NULL;
		tok = strtok(line, " \n"); strtok(NULL, " \n");
		nNodes[i]=atoi(strtok(NULL, " \n"));
		treePts[i]->setNNodes(nNodes[i]);
		treePts[i]->buildTreeFromFile(input,treePts[i]->nodePts[0]);
	}
	input.close();
	return 0;
}

/*copy training data leaf distribution to oob leaf distribution. Invoked if oob=0 */
void BudgetForest::copyOob(){
	for(int i=0;i<nTrees;i++)
		treePts[i]->copyOob();
}

/*classify dataset test to obtain predictions*/
void BudgetForest::classify(const vec_data_t& test, args_t& args, vector<vector<int>>& pred1,vector<vector<int>>& pred2,vector<vector<int>>& pred3,vector<vector<double>>& pred1Rank,vector<vector<double>>& pred2Rank ,vector<vector<double>>& pred3Rank , vector<double>& errPrec1, vector<double>& errPrec2,vector<double>& errPrec3,vector<double>& costE){
	int numthreads=args.processors,i, t,j,k;
	int fk = args.kfeatures;
	int max_c=0,max_c2=0;
	int nfeatures=args.features;
	
	if (numthreads > nTrees)
		numthreads = nTrees;
	nTrees = (nTrees / numthreads) * numthreads;

	//evaluate all the trees:
	vector<bool**> testFeatureUsed; //num of datasets, num of data examples, num of features
	for (i=0;i<test.size();i++){ //for each test dataset
		bool** featDataset= new bool*[test[i].size()]; //, vector<bool>(nfeatures,false));
		for(j=0;j<test[i].size();j++){
			featDataset[j] = new bool[nfeatures];
		}
		testFeatureUsed.push_back(featDataset);
	}
	for (i=0;i<test.size();i++) //for each test dataset
		for(j=0;j<test[i].size();j++)
			for(k=0;k<nfeatures;k++)
				testFeatureUsed[i][j][k]=false;

	for (i=0;i<test.size();i++){ //for each test dataset
		int nte=test[i].size();
		struct PredOutputs outStruct;
		outStruct.pred1_tmp.resize(nte,0);
		outStruct.pred2_tmp.resize(nte,0);
		outStruct.pred3_tmp.resize(nte,0);
		outStruct.pred1Rank_tmp.resize(nte,0.0);
		outStruct.pred2Rank_tmp.resize(nte,0.0);
		outStruct.pred3Rank_tmp.resize(nte,0.0);
		int num_test_per_p=nte/numthreads;
		fprintf(stderr, "About to evaluate trees using threading\n");
		thread** threads = new thread*[numthreads];
		for (t=0;t<numthreads;t++){
			threads[t] = new thread(boost::bind(&BudgetForest::classifyInRange, this, test[i],  boost::cref(args), t*nte/numthreads, (t+1)*nte/numthreads, boost::ref(outStruct), testFeatureUsed[i]));
		}
		for (j=0;j<numthreads;j++){
			threads[j]->join();
			delete threads[j];
		}
		fprintf(stderr, "done threading\n");
		delete[] threads;
		//make a copy of outStruct

		pred1.push_back(outStruct.pred1_tmp);
		pred2.push_back(outStruct.pred2_tmp);
		pred3.push_back(outStruct.pred3_tmp);
		pred1Rank.push_back(outStruct.pred1Rank_tmp);
		pred2Rank.push_back(outStruct.pred2Rank_tmp);
		pred3Rank.push_back(outStruct.pred3Rank_tmp);
	}
	//compute classification error
	for(i=0;i<test.size();i++){ //for each test dataset
		for (j=0;j<test[i].size();j++){ //for each test example in the ith dataset
			if (test[i][j]->label != pred1[i][j])
				errPrec1[i]++;
			if (test[i][j]->label != pred2[i][j])
				errPrec2[i]++;
			if (test[i][j]->label != pred3[i][j])
				errPrec3[i]++;
		}
		errPrec1[i]=errPrec1[i]/test[i].size();
		errPrec2[i]=errPrec2[i]/test[i].size();
		errPrec3[i]=errPrec3[i]/test[i].size();
	}
	// aggregate feature usage matrix
	for (i=0;i<test.size();i++){
		for(j=0;j<test[i].size();j++){
			for (k=1;k<nfeatures;k++){
				if (testFeatureUsed[i][j][k]==true) costE[i]+=args.Costs[k];
			}
		}
		costE[i]=costE[i]/test[i].size();
	}
		//delete testFeatureUsed
	for (i=0;i<test.size();i++){ //for each test dataset
		for(j=0;j<test[i].size();j++)
			delete[] testFeatureUsed[i][j];
		delete[] testFeatureUsed[i];
	}
}

/*classify subset of dataset test */
void BudgetForest::classifyInRange(data_t test,  const args_t& args, int test_start, int test_end, struct PredOutputs& outStruct, bool** testFeatureUsed){
		std::ostringstream oss;
		oss << boost::this_thread::get_id();
		std::string idAsString = oss.str();
		cout<<  "thread id:" <<idAsString<<endl;
		std::seed_seq seed1(idAsString.begin(), idAsString.end());
		std::default_random_engine gen(seed1);

		int i,k, t1,t2, sum_c1,sum_c2,sum_c3,max_c1,max_c2,max_c3, maxTmp,indTmp;
		double v = 0.0;
		BudgetNode* curNode;
		tupleW* instance;
		int f=0;
		int useOOB=args.oob; //if evaluate pruned==1 trees we can use Oob_C_leaf instead of c_leaf

		for (i=test_start; i<test_end; i++) { //for each test example in range
			boost::numeric::ublas::vector<int> c1(numberOfClasses,0);
			boost::numeric::ublas::vector<int> c2(numberOfClasses,0);			
			boost::numeric::ublas::vector<int> c3(numberOfClasses,0);			
			for(t1=0;t1<nTrees;t1++){
				curNode=treePts[t1]->nodePts[0]; //curNode points to the root of tree
				instance=test[i];
				while(!curNode->leaf){
					f=curNode->feature;
					testFeatureUsed[i][f]=true;
					v=curNode->value;
					if (instance->features[f] == UNKNOWN){
						if (curNode->child[BudgetNode::MISSING]==0){
							break;
						}
						else{
							curNode=curNode->child[BudgetNode::MISSING];
						}
					}
					else if (instance->features[f]<=v){
						curNode=curNode->child[BudgetNode::YES];
					}
					else
						curNode=curNode->child[BudgetNode::NO];
				}
				c1[curNode->pred]++;
				maxTmp=0;
				indTmp=0;
				for (t2=0;t2<numberOfClasses;t2++){
					if (curNode->oob_c_leaf[t2]>maxTmp){
						maxTmp=curNode->oob_c_leaf[t2];
						indTmp=t2;
					}
				}
				c2[indTmp]++;
				c3+=curNode->c_leaf;		
			}
			max_c1=0;
			max_c2=0;
			max_c3=0;
			sum_c1=0;
			sum_c2=0;
			sum_c3=0;
			for (k=0;k<numberOfClasses;k++){
				sum_c1+=c1[k];
				sum_c2+=c2[k];
				sum_c3+=c3[k];
				if (c1[k]>max_c1){
					max_c1=c1[k];
					outStruct.pred1_tmp[i]=k;
				}
				if (c2[k]>max_c2){
					max_c2=c2[k];
					outStruct.pred2_tmp[i]=k;
				}
				if (c3[k]>max_c3){
					max_c3=c3[k];
					outStruct.pred3_tmp[i]=k;
				}
				outStruct.pred1Rank_tmp[i]+=k*c1[k];
				outStruct.pred2Rank_tmp[i]+=k*c2[k];
				outStruct.pred3Rank_tmp[i]+=k*c3[k];
			}
			outStruct.pred1Rank_tmp[i]=outStruct.pred1Rank_tmp[i]/sum_c1;
			outStruct.pred2Rank_tmp[i]=outStruct.pred2Rank_tmp[i]/sum_c2;
			outStruct.pred3Rank_tmp[i]=outStruct.pred3Rank_tmp[i]/sum_c3;
		}
}

/*write the trees in text format*/
void BudgetForest::writeTrees(char* treesFileName){
	int i;
	ofstream outfile;
	outfile.open(treesFileName);
	for(i=0;i<nTrees;i++){
		outfile<< "Tree "<< i <<" "<<nNodes[i]<<endl;
		treePts[i]->writeTree(treePts[i]->nodePts[0],outfile);
	}
	outfile.close();
}

BudgetForest::~BudgetForest(){
	//delete trees
	for (int i=0;i<nTrees;i++)
			delete treePts[i];
}

/*build nTreesPerProc trees, for current processor*/
void BudgetForest::buildPerProc(int nTreesPerProc, const data_t& train, const vec_data_t& test, args_t& args, vector<BudgetTree*>& treesPerProc){
	std::ostringstream oss;
	oss << boost::this_thread::get_id();
	std::string idAsString = oss.str();
	cout<<  "thread id:" <<idAsString<<endl;
	std::seed_seq seed1(idAsString.begin(), idAsString.end());
	std::default_random_engine gen(seed1);
	int t,num_c=args.num_c, num_test=test.size();
	double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double) = NULL;
	if (args.loss==ALG_HP)
		impurityHandle= impurityHP;
	else if(args.loss==ALG_ENTROPY){
		impurityHandle= impurityEntropy;
	}
	else{
		fprintf(stderr, "Only support HP and Entropy for now!\n");
		return;
	}
	for (t = 0; t < nTreesPerProc; t++) {
		treesPerProc[t]=new BudgetTree(1,numberOfClasses);
		treesPerProc[t]->buildLearn(train, test, args,impurityHandle,gen);
	}
}

/*build nTreesPerProc trees, for current processor*/
void BudgetForest::buildLearn(const data_t& train, const vec_data_t& test, args_t& args){
	int numthreads=args.processors, i,j,k;
	int fk = args.kfeatures;
	int max_c=0,max_c2=0;
	int nfeatures=args.features;

	if (numthreads > nTrees)
		numthreads = nTrees;
	nTrees = (nTrees / numthreads) * numthreads;

	int nTreesPerProc = nTrees / numthreads;  
	thread** threads = new thread*[numthreads];


	vector<vector<BudgetTree*>> treesPerProc(numthreads, vector<BudgetTree*>(nTreesPerProc,NULL));
	for (i=0;i<numthreads;i++)
		threads[i] = new thread(boost::bind(&BudgetForest::buildPerProc, this, nTreesPerProc, boost::cref(train), boost::cref(test),  boost::ref(args), boost::ref(treesPerProc[i])));
  
	for (i=0;i<numthreads;i++){
		threads[i]->join();
		delete threads[i];
	}
	fprintf(stderr, "done threading\n");
	delete[] threads;

	//vectorize the trees for easy evaluation
	k=0;
	for(i=0;i<numthreads;i++){
		for(j=0;j<nTreesPerProc;j++){
			treePts[k]=treesPerProc[i][j];
			k++;
		}
	}
	for(i=0;i<nTrees;i++)
		nNodes[i]=treePts[i]->nNodes;
}
