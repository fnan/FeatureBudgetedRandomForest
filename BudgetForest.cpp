#include "BudgetForest.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

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
		treePts[i]=new BudgetTree(nNodes[i],numberOfClasses);
		treePts[i]->prunedAway=atof(strtok(NULL, " \n"));
		treePts[i]->buildTreeFromFile(input,treePts[i]->nodePts[0]);
	}
	input.close();
	return 0;
}

/* construct trees from treesFileName */
int BudgetForest::readTreesFromFileIncre(char* treesFileName, int start, int incre){
	int i,j;
	ifstream input(treesFileName);
	if (input.fail())
		return 1;

	string strline;
	int curTree=0,nNodesTmp;
	/*skip until the tree index: start*/
	for(i=0;i<nTrees;i++){
		if(curTree>=start)
			break;
		getline(input, strline);
		char* line = _strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
		char* tok = NULL;
		tok = strtok(line, " \n"); 
		if(strcmp(tok, "Tree")){
			cout<<"Error reading tree incrementally in readTreesFromFileIncre:" << treesFileName <<endl;
			return 1;
		}
		strtok(NULL, " \n");
		nNodesTmp=atoi(strtok(NULL, " \n"));
		for(j=0;j<nNodesTmp;j++)
			getline(input,strline);

		curTree++;
	}
	for(i=0;i<incre;i++){		
		getline(input, strline);
		char* line = _strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
		char* tok = NULL;
		tok = strtok(line, " \n"); strtok(NULL, " \n");
		nNodes[i]=atoi(strtok(NULL, " \n"));
		treePts[i]=new BudgetTree(nNodes[i],numberOfClasses);
		treePts[i]->prunedAway=atof(strtok(NULL, " \n"));
//		treePts[i]->setNNodes(nNodes[i]);
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

/* count the number of times each feature appears in each tree */
void BudgetForest::featureTreeCount(args_t& args,int r){
	int i,j,count;
	char featureTreeFileName[100];
	sprintf(featureTreeFileName, "%s_%d_%d_%f_featTreCnt",args.train_file,r, args.trees,(args.prune));
	ofstream outfile;
	outfile.open(featureTreeFileName);

	for(i=0;i<treePts.size();i++){
		vector<int> featureCount(args.features,0);
		for(j=0;j<treePts[i]->nodePts.size();j++){
			if(treePts[i]->nodePts[j]->stopP<0.99)
				featureCount[treePts[i]->nodePts[j]->feature]++;
		}
		for(j=1;j<featureCount.size()-1;j++)
			outfile<<featureCount[j]<<" ";
		outfile<<featureCount[featureCount.size()-1]<<endl;
	}
	outfile.close();
}
/*classify dataset test to obtain predictions*/
void BudgetForest::classify(const vec_data_t& test, args_t& args, vector<vector<int>>& pred1,vector<vector<int>>& pred2,vector<vector<int>>& pred3,vector<vector<double>>& pred1Rank,vector<vector<double>>& pred2Rank ,vector<vector<double>>& pred3Rank , vector<double>& errPrec1, vector<double>& errPrec2,vector<double>& errPrec3,vector<double>& costE, int r){
	int numthreads=args.processors,i, t,j,k;
	int max_c=0,max_c2=0;
	
	if (numthreads > nTrees)
		numthreads = nTrees;
	//nTrees = (nTrees / numthreads) * numthreads;

	//evaluate all the trees:
	vector<int**> testFeatureUsed; //num of datasets, num of data examples, num of features
	for (i=0;i<test.size();i++){ //for each test dataset
		int** featDataset= new int*[test[i].size()]; //, vector<bool>(nfeatures,false));
		for(j=0;j<test[i].size();j++){
			featDataset[j] = new int[args.sensors];
		}
		testFeatureUsed.push_back(featDataset);
	}
	for (i=0;i<test.size();i++) //for each test dataset
		for(j=0;j<test[i].size();j++)
			for(k=0;k<args.sensors;k++)
				testFeatureUsed[i][j][k]=0;

	vector<vector<vector<int>>> leafIndexAll;
	for (i=0;i<test.size();i++){ //for each test dataset
		int nte=test[i].size();
		vector<vector<double>> proba_pred(nte, vector<double>(args.num_c, 0.0)); //store the predicted class probabilities

		struct PredOutputs outStruct;
		vector<vector<int>> leafIndexTmp(nte, vector<int>(nTrees,0));
		outStruct.proba_pred = proba_pred;
		outStruct.leafIndex=leafIndexTmp;
		outStruct.pred1_tmp.resize(nte,0);
		outStruct.pred2_tmp.resize(nte,0);
		outStruct.pred3_tmp.resize(nte,0);
		outStruct.pred1Rank_tmp.resize(nte,0.0);
		outStruct.pred2Rank_tmp.resize(nte,0.0);
		outStruct.pred3Rank_tmp.resize(nte,0.0);
		int num_test_per_p=nte/numthreads;

		threadingRange threadingInp;
		threadingInp.dataInp = test[i];


		fprintf(stderr, "About to evaluate trees using threading\n");
		thread** threads = new thread*[numthreads];
		for (t=0;t<numthreads;t++){
			threadingInp.start = t*nte / numthreads;
			threadingInp.end = (t + 1)*nte / numthreads;
			threads[t] = new thread(&BudgetForest::classifyInRange, this, threadingInp, cref(args), ref(outStruct), testFeatureUsed[i]);
		}
		for (j=0;j<numthreads;j++){
			threads[j]->join();
			delete threads[j];
		}
		fprintf(stderr, "done threading\n");
		delete[] threads;
		//make a copy of outStruct
		leafIndexAll.push_back(outStruct.leafIndex);
		pred1.push_back(outStruct.pred1_tmp);
		pred2.push_back(outStruct.pred2_tmp);
		pred3.push_back(outStruct.pred3_tmp);
		pred1Rank.push_back(outStruct.pred1Rank_tmp);
		pred2Rank.push_back(outStruct.pred2Rank_tmp);
		pred3Rank.push_back(outStruct.pred3Rank_tmp);

		//output the prediction probabilities to file:
		char prob_pred_fileName[100];
		sprintf(prob_pred_fileName, "%s_%d_%d_%f_proba_pred_%d", args.train_file, r, args.trees, (args.prune), i);
		ofstream outfile;
		outfile.open(prob_pred_fileName);
		for (j = 0; j < nte; j++){
			for (k = 0; k < args.num_c; k++)
				outfile << outStruct.proba_pred[j][k] << " ";
			outfile << endl;
		}
		outfile.close();

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
			for (k=1;k<args.sensors;k++){
				if (testFeatureUsed[i][j][k]>0) costE[i]+=args.CostSensors[k];
			}
		}
		costE[i]=costE[i]/test[i].size();
	}
	
	/*save testFeatureUsed if verbose>=2 */
//	if(args.verbose>=2){
	for (i = 0; i < test.size(); i++){ //for each test dataset
		char featureUsedFileName[100];
		sprintf(featureUsedFileName, "%s_%d_%d_%f_featMatrix_%d", args.train_file, r, args.trees, (args.prune), i);
		ofstream outfile;
		outfile.open(featureUsedFileName);
		for (j = 0; j < test[i].size(); j++){
			for (k = 1; k < args.sensors - 1; k++){
				outfile << testFeatureUsed[i][j][k] << " ";
			}
			outfile << testFeatureUsed[i][j][k] << endl;
		}
		outfile.close();
	}
//	}

	/*save leaf index if boosting*/
	if(args.alg==ALG_BOOST_MAXSPLIT || args.alg==ALG_BOOST_EXPSPLIT){
		char leafIndexFileName[100];
		for(k=0;k<test.size();k++){
			if(args.alg==ALG_BOOST_EXPSPLIT)
				sprintf(leafIndexFileName, "%sEXP_%d_%f_d%dleafIndex%d",args.train_file, args.trees,(args.prune),args.depth,k);
			if(args.alg==ALG_BOOST_MAXSPLIT)
				sprintf(leafIndexFileName, "%sMAX_%d_%f_d%dleafIndex%d",args.train_file, args.trees,(args.prune),args.depth,k);
			ofstream outfile;
			outfile.open(leafIndexFileName);
			for(i=0;i<test[k].size();i++){
				for(j=0;j<nTrees;j++)
					outfile<<leafIndexAll[k][i][j]<<" ";
				outfile<<pred1[k][i]<<endl;
			}
			outfile.close();
		}
	}
	//delete testFeatureUsed
	for (i=0;i<test.size();i++){ //for each test dataset
		for(j=0;j<test[i].size();j++)
			delete[] testFeatureUsed[i][j];
		delete[] testFeatureUsed[i];
	}
}

/*classify subset of dataset test */
void BudgetForest::classifyInRange(threadingRange threadingInp, const args_t& args, struct PredOutputs& outStruct, int** testFeatureUsed){
		
	data_t test = threadingInp.dataInp;
	int test_start = threadingInp.start;
	int test_end = threadingInp.end;

		std::ostringstream oss;
		oss << this_thread::get_id();
		std::string idAsString = oss.str();
		//cout<<  "thread id:" <<idAsString<<endl;
		std::seed_seq seed1(idAsString.begin(), idAsString.end());
		std::default_random_engine gen(seed1);

		int i,k, t1,t2, sum_c1,sum_c2,max_c1,max_c2, maxTmp,indTmp,sumTmpOOB,sumTmpC_leaf;
		double v = 0.0,max_c3,sum_c3;
		BudgetNode* curNode;
		tupleW* instance;
		int f=0;
		bool pruned=(args.prune>=0);
		int useOOB=args.oob; //if evaluate pruned==1 trees we can use Oob_C_leaf instead of c_leaf
		std::uniform_real_distribution<double> dist(0.0,1.0);

			for (i=test_start; i<test_end; i++) { //for each test example in range
				vector<int> c1(numberOfClasses,0);
				vector<int> c2(numberOfClasses,0);			
				vector<double> c3(numberOfClasses,0);			
				if(pruned){
					for(t1=0;t1<nTrees;t1++){
						if(treePts[t1]->prunedAway>0.999) //if pruned away, do not consider this tree
							continue;
						curNode=treePts[t1]->nodePts[0]; //curNode points to the root of tree
						instance=test[i];
						while((!curNode->leaf) && (dist(gen)>=curNode->stopP)){//stop if it's a stop node by pruning
							f=curNode->feature;
							testFeatureUsed[i][args.Costgroup[f]]++;
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
						//c2 aggregates absolute margin
						for(t2=0;t2<numberOfClasses;t2++){
							c2[t2]+=curNode->c_leaf[t2];
						}
						//maxTmp=0;
						//indTmp=0;
						//sumTmpOOB=0;
						sumTmpC_leaf=0;
						for (t2=0;t2<numberOfClasses;t2++){
							//sumTmpOOB+=curNode->oob_c_leaf[t2];
							sumTmpC_leaf+=curNode->c_leaf[t2];
							//if (curNode->oob_c_leaf[t2]>maxTmp){
							//	maxTmp=curNode->oob_c_leaf[t2];
							//	indTmp=t2;
							//}
						}
						//c2[indTmp]++;
						/* oob shouldn't be used to predict test point label
						if(useOOB==2 && sumTmpOOB>0){
							for(t2=0;t2<numberOfClasses;t2++)
								c3[t2]+=curNode->oob_c_leaf[t2]/(double)sumTmpOOB;
						}
						else */
						if(sumTmpC_leaf>0){
							for(t2=0;t2<numberOfClasses;t2++)
								c3[t2]+=curNode->c_leaf[t2]/(double)sumTmpC_leaf;
						}
					}		
				}else{
					for(t1=0;t1<nTrees;t1++){
						curNode=treePts[t1]->nodePts[0]; //curNode points to the root of tree
						instance=test[i];
						while(!curNode->leaf){
							f=curNode->feature;
							testFeatureUsed[i][args.Costgroup[f]]++;
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
						//c2 aggregates absolute margin
						for(t2=0;t2<numberOfClasses;t2++){
							c2[t2]+=curNode->c_leaf[t2];
						}
						//maxTmp=0;
						//indTmp=0;
						//sumTmpOOB=0;
						sumTmpC_leaf=0;
						for (t2=0;t2<numberOfClasses;t2++){
						//	sumTmpOOB+=curNode->oob_c_leaf[t2];
							sumTmpC_leaf+=curNode->c_leaf[t2];
						//	if (curNode->oob_c_leaf[t2]>maxTmp){
						//		maxTmp=curNode->oob_c_leaf[t2];
						//		indTmp=t2;
						//	}
						}
						//c2[indTmp]++;
						/* oob shouldn't be used to predict test point label
						if(useOOB==2 && sumTmpOOB>0){
							for(t2=0;t2<numberOfClasses;t2++)
								c3[t2]+=curNode->oob_c_leaf[t2]/(double)sumTmpOOB;
						}
						else */
						if(sumTmpC_leaf>0){
							for(t2=0;t2<numberOfClasses;t2++)
								c3[t2]+=curNode->c_leaf[t2]/(double)sumTmpC_leaf;
						}
					}
				}
				max_c1=0;
				max_c2=0;
				max_c3=0.0;
				sum_c1=0;
				sum_c2=0;
				sum_c3=0.0;
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
				for (k = 0; k < numberOfClasses; k++)
					outStruct.proba_pred[i][k] = c3[k] / sum_c3;

				outStruct.pred1Rank_tmp[i]=outStruct.pred1Rank_tmp[i]/sum_c1;
				outStruct.pred2Rank_tmp[i]=outStruct.pred2Rank_tmp[i]/sum_c2;
				outStruct.pred3Rank_tmp[i]=outStruct.pred3Rank_tmp[i]/sum_c3;
			}
		//}
}

/*write the trees in text format*/
void BudgetForest::writeTrees(char* treesFileName, int append){
	int i;
	ofstream outfile;
	if(append==0)
		outfile.open(treesFileName);
	else
		outfile.open(treesFileName, std::ofstream::out | std::ofstream::app);

	for(i=0;i<nTrees;i++){
		outfile<< "Tree "<< i <<" "<<nNodes[i]<<" "<<treePts[i]->prunedAway<< endl;
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
void BudgetForest::buildPerProc(int nTreesPerProc, const data_t& train, args_t& args, vector<BudgetTree*>& treesPerProc){
	std::ostringstream oss;
	oss << this_thread::get_id();
	std::string idAsString = oss.str();
	cout<<  "thread id:" <<idAsString<<endl;
	std::seed_seq seed1(idAsString.begin(), idAsString.end());
	std::default_random_engine gen(seed1);
	int t, num_c = args.num_c;
	double (*impurityHandle)(int, vector<int>&,double) = NULL;
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
		treesPerProc[t]->buildLearn(train, args,impurityHandle,gen);
	}
}

/*build nTreesPerProc trees, for current processor*/
void BudgetForest::buildLearn( data_t& train, args_t& args){
	int numthreads=args.processors,i,j,k;
	int fk = args.kfeatures;
	int max_c=0,max_c2=0;
	int nfeatures=args.features;
	/*vector<double> currentPred(train.size(),0.0);*/

		if (numthreads > nTrees)
			numthreads = nTrees;
		nTrees = (nTrees / numthreads) * numthreads;

		int nTreesPerProc = nTrees / numthreads;  
		thread** threads = new thread*[numthreads];


		vector<vector<BudgetTree*>> treesPerProc(numthreads, vector<BudgetTree*>(nTreesPerProc,NULL));
		for (i=0;i<numthreads;i++)
			threads[i] = new thread(&BudgetForest::buildPerProc, this, nTreesPerProc, cref(train), ref(args), ref(treesPerProc[i]));
  
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


/*return the number of internal nodes in the forest after pruning, based on stopP */
int BudgetForest::numInternalNodes(){
	int i,j,count=0;
	for(i=0;i<nTrees;i++){
		for(j=0;j<nNodes[i];j++){
			if(treePts[i]->nodePts[j]->stopP<0.5)
				count++;
		}
	}
	return count;
}

void BudgetForest::updateExampleFeatCount(int treeID, int curNodeID, const vector<vector<vector<int>>>& examplesInNode, vector<vector<int>>& exampleFeatCount){
	if(treePts[treeID]->nodePts[curNodeID]->stopP==1 || treePts[treeID]->nodePts[curNodeID]->leaf) return;

	for(int i=0;i<examplesInNode[treeID][curNodeID].size();i++){
		exampleFeatCount[examplesInNode[treeID][curNodeID][i]][treePts[treeID]->nodePts[curNodeID]->feature]--;
	}
	updateExampleFeatCount(treeID, treePts[treeID]->nodePts[curNodeID]->child[0]->ID, examplesInNode, exampleFeatCount);
	updateExampleFeatCount(treeID, treePts[treeID]->nodePts[curNodeID]->child[1]->ID, examplesInNode, exampleFeatCount);
}

void BudgetForest::fillExampleFeatCountInRange(const data_t& train, const args_t& myargs, int test_start, int test_end, vector<vector<int>>& exampleFeatCount){
	int i, j, f;
	tupleW* instance;
	BudgetNode* curNode;
	for (i = test_start; i<test_end; i++){
		for (j = 0; j<myargs.features; j++)
			exampleFeatCount[i][j] = 0;
		instance = train[i];
		for (j = 0; j<nTrees; j++){
			curNode = treePts[j]->nodePts[0]; //curNode points to the root of tree					
			while (curNode->stopP != 1 && !curNode->leaf){
				f = curNode->feature;
				exampleFeatCount[i][f]++;
				if (instance->features[f] <= curNode->value)
					curNode = curNode->child[BudgetNode::YES];
				else
					curNode = curNode->child[BudgetNode::NO];
			}
		}
	}
}


/*write the output tree from CCP*/
void BudgetForest::writeCCP(char* treesFileName){
	int i, curID;
	ofstream outfile;
	outfile.open(treesFileName);
	for (i = 0; i<nTrees; i++){
		curID = 0;
		treePts[i]->reassignID(treePts[i]->nodePts[0], &curID, -1);
		outfile << "Tree " << i << " " << curID << " " << treePts[i]->prunedAway << endl;
		treePts[i]->writePrunedTree(treePts[i]->nodePts[0], outfile);
	}
	outfile.close();
}


void BudgetForest::avgprec2Simple(const data_t& data, vector<double>& predRank, int topX, double& prec){
	int n=data.size(), i,j, ind;
	int nq=0;
	double avg=0.0;
	i=0;
	while (i<n){
		int q=data[i]->qid;
		vector<pair<double, int>> tmp;
		for(j=0;j<n;j++)
			if(data[j]->qid ==q){
				tmp.push_back(pair<double, int>(predRank[j], data[j]->label));
			}
			std::sort(tmp.begin(), tmp.end(), [](const pair<double, int> &left, const pair<double, int> &right){ return left.first > right.first;  });
		ind=0;
		while(ind<topX && ind<tmp.size() &&  tmp[ind].second==1) ind++;
		if(ind==tmp.size())
			avg+=1.0;
		else
			avg+=(double)ind/topX;

		i+=tmp.size();
		nq++;
	}
	prec=avg/nq;
}
