#include <iostream>
#include <fstream>
#include <string>
#include "BudgetTree.h"
using namespace std;

/*constructor*/
BudgetTree::BudgetTree(int nNodes, int numberOfClasses){
	this->nNodes=nNodes;
	this->numberOfClasses=numberOfClasses;
	this->nodePts.resize(nNodes,NULL);
	this->prunedAway=0;
}


/* construct trees from treesFileName */
int BudgetTree::buildTreeFromFile(ifstream& input, BudgetNode*& parNode){
	int i;
	string strline;
	getline(input, strline);
	char* line = _strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
	char* tok = NULL;
	tok = strtok(line, " \n");

	parNode= new BudgetNode(numberOfClasses);
	parNode->ID=atoi(tok);
	parNode->parID=atoi(strtok(NULL, " \n"));
	strtok(NULL, " \n");//discard YesID
	strtok(NULL, " \n");//discard NoID
	parNode->feature=atoi(strtok(NULL, " \n"));
	parNode->value=atof(strtok(NULL, " \n"));
	parNode->leaf=(atoi(strtok(NULL, " \n"))==1);
	parNode->stopP=atof(strtok(NULL, " \n"));
	parNode->pred=atoi(strtok(NULL, " \n"));
	parNode->pred_double=atof(strtok(NULL, " \n"));
	for(i=0;i<numberOfClasses;i++)
		parNode->c_leaf[i]=atoi(strtok(NULL, " \n"));
	for(i=0;i<numberOfClasses;i++)
		parNode->oob_c_leaf[i]=atoi(strtok(NULL, " \n"));
	//record the node pointers
	nodePts[parNode->ID]=parNode;
	if(parNode->leaf){
		return 0;
	}
	buildTreeFromFile(input,parNode->child[0]);
	buildTreeFromFile(input,parNode->child[1]);
}

/*copy training data leaf distribution to oob leaf distribution. Invoked if oob=0 */
void BudgetTree::copyOob(){
	for(int i=0;i<nNodes;i++)
		nodePts[i]->oob_c_leaf=nodePts[i]->c_leaf;
}
/*reassign node ID on pruned tree based on stopP */
void BudgetTree::reassignID(BudgetNode* curNode, int* curID, int parentID){
	int ID;
	if(curNode==0) return;
	ID=*curID;
	curNode->ID=*curID;
	curNode->parID=parentID;
	*curID=ID+1;
	if(curNode->stopP==1 || curNode->leaf) return;
	reassignID(curNode->child[0], curID, ID);
	reassignID(curNode->child[1], curID, ID);
}
/*must follow reassignID()*/
void BudgetTree::writePrunedTree(const BudgetNode* curNode, ofstream& outfile){
	if(curNode==0) return;
	int i;
	int YesID=-1, NoID=-1;
	if(curNode->stopP==1 || curNode->leaf){
		//each line: NodeID parID YESID NOID featureID splitValue isLeaf stopP pred [c_leaf] [oob_c_leaf]
		outfile<< curNode->ID << " "<< curNode->parID <<" "<<YesID << " " << NoID << " " << curNode->feature << " " << curNode->value << " " << 1 << " " <<curNode->stopP <<" "<< curNode->pred <<" ";
		for(i=0;i<curNode->c_leaf.size();i++)
			outfile<<curNode->c_leaf[i]<<" ";
		for(i=0;i<curNode->oob_c_leaf.size();i++)
			outfile<<curNode->oob_c_leaf[i]<<" ";
		outfile<<endl;
		return;
	}
	if(!curNode->leaf){
		YesID=curNode->child[0]->ID;
		NoID=curNode->child[1]->ID;
	}
	//each line: NodeID parID YESID NOID featureID splitValue isLeaf stopP pred [c_leaf] [oob_c_leaf]
	outfile<< curNode->ID << " "<< curNode->parID <<" "<<YesID << " " << NoID << " " << curNode->feature << " " << curNode->value << " " << curNode->leaf << " " <<curNode->stopP <<" "<< curNode->pred <<" ";
	for(i=0;i<curNode->c_leaf.size();i++)
		outfile<<curNode->c_leaf[i]<<" ";
	for(i=0;i<curNode->oob_c_leaf.size();i++)
		outfile<<curNode->oob_c_leaf[i]<<" ";
	outfile<<endl;
	writePrunedTree(curNode->child[0], outfile);
	writePrunedTree(curNode->child[1],outfile);

}
void BudgetTree::writeTree(const BudgetNode* curNode, ofstream& outfile){
	if(curNode==0) return;
	int i;
	int YesID=-1, NoID=-1;
	if(!curNode->leaf){
		YesID=curNode->child[0]->ID;
		NoID=curNode->child[1]->ID;
	}
	//each line: NodeID parID YESID NOID featureID splitValue isLeaf stopP pred [c_leaf] [oob_c_leaf]
	outfile<< curNode->ID << " "<< curNode->parID<<" "<<YesID << " " << NoID << " " << curNode->feature << " " << curNode->value << " " << curNode->leaf << " " <<curNode->stopP <<" "<< curNode->pred <<" "<< curNode->pred_double <<" ";
	for(i=0;i<curNode->c_leaf.size();i++)
		outfile<<curNode->c_leaf[i]<<" ";
	for(i=0;i<curNode->oob_c_leaf.size();i++)
		outfile<<curNode->oob_c_leaf[i]<<" ";
	outfile<<endl;
	writeTree(curNode->child[0], outfile);
	writeTree(curNode->child[1],outfile);
}
void BudgetTree::buildLearn(const data_t& train, args_t& args, double (*impurityHandle)(int, vector<int>&,double),std::default_random_engine& gen){
	int kfeatures = args.kfeatures;
//	int num_test = test.size();
	int i, ID=0;
	data_t sample;
	std::uniform_int_distribution<int> dist(0,train.size()-1);
	for (i=0; i < ceil(train.size()*args.bagging_rate); i++)
		sample.push_back(train[dist(gen)]);

	////quick check for sample randomness:
	//int indexSum=0;
	//for (i=0;i<sample.size();i++)
	//	indexSum+=sample[i]->idx;
	//cout<<  "random sampling indexSum= " <<indexSum<<endl;
	nodePts[0] = new BudgetNode(sample, args, 1, impurityHandle, &ID, -1,gen);
	nNodes=ID;
	updateNodePts();
	if(args.oob>=1){ //include oob samples
		oobUpdate(train,sample); //update leaf distribution using oob samples: oob_c_leaf
		//ValUpdate(test[0]); //update leaf distribution using validation samples
	}
}

/*update nodePts for this tree based on nNodes*/
void BudgetTree::updateNodePts(){
	nodePts.resize(nNodes,NULL);
	BudgetNode* curNode=nodePts[0]; //root node
	traverseTree(curNode);
}

/*pre-order traversal of the tree*/
void BudgetTree::traverseTree(BudgetNode* curNode){
	nodePts[curNode->ID]=curNode;
	if(curNode->leaf)
		return;
	traverseTree(curNode->child[0]);
	traverseTree(curNode->child[1]);
}

/*update leaf distribution using oob samples*/
void BudgetTree::oobUpdate(const data_t& train, const data_t& sample){
	int i,k,f;
	int n=train.size();
	BudgetNode* curNode;
	tupleW* instance;
	vector<int> indx(sample.size(),-1);
	double v;
	for(i=0;i<sample.size();i++){
		indx[i]=sample[i]->idx;
	}
	sort(indx.begin(),indx.end());
	for(i=0;i<n;i++){
		k=0;
		while(k<n && i>indx[k])k++;
		if(k>=n || i!=indx[k]){
			// data i is out-of-bag, evaluate it on the tree
			curNode=nodePts[0]; //curNode points to the root of tree
			instance=train[i];
			while(!curNode->leaf){
				curNode->oob_c_leaf[instance->label]++;
				f=curNode->feature;
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
			curNode->oob_c_leaf[instance->label]++;
		}
	}
}
 //update leaf distribution using validation samples
void BudgetTree::ValUpdate(const data_t& val){
	int i,j,f;
	int n=val.size();
	double v;
	BudgetNode* curNode;
	tupleW* instance;
	for(i=0;i<n;i++){
			// evaluate data i on the tree
		curNode=nodePts[0]; //curNode points to the root of tree
			instance=val[i];
			while(!curNode->leaf){
				curNode->oob_c_leaf[instance->label]++;
				f=curNode->feature;
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
			curNode->oob_c_leaf[val[i]->label]++;	
	}
}

void BudgetTree::predSimple(const data_t& data, const args_t& args, vector<double>& pred){
	int numthreads=args.processors,i, t,j,k;	
	//evaluate current tree:
	int nt=data.size();
	int num_test_per_p=nt/numthreads;
//	fprintf(stderr, "About to evaluate trees using threading\n");
	std::thread** threads = new std::thread*[numthreads];
	threadingRange rangeInd;

	for (t=0;t<numthreads;t++){
		rangeInd.start = t*nt / numthreads;
		rangeInd.end=(t+1)*nt/numthreads;
		threads[t] = new thread(&BudgetTree::predInRangeSimple, this, std::cref(data),  rangeInd,  std::ref(pred));
	}
	for (j=0;j<numthreads;j++){
		threads[j]->join();
		delete threads[j];
	}
//	fprintf(stderr, "done threading\n");
	delete[] threads;
}

void BudgetTree::predInRangeSimple(const data_t& data, threadingRange& rangeInd, vector<double>& pred){
	int i,k, t1,t2;
	double v = 0.0,max_c3,sum_c3;
	BudgetNode* curNode;
	tupleW* instance;
	int f=0;
	int test_start=rangeInd.start, test_end=rangeInd.end;

	for (i=test_start; i<test_end; i++) { //for each test example in range
		curNode=nodePts[0]; //curNode points to the root of tree
		instance=data[i];
		while(!curNode->leaf && curNode->stopP==0.0){
			f=curNode->feature;
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
		pred[i]=curNode->pred_double;
	}

}

void BudgetTree::updateLeafindex(const data_t& data){
	leafIndex.resize(nNodes, -1);
	leafPerExample.resize(data.size(),-1);

	int leafCount=0, f;
	tupleW* instance;
	BudgetNode* curNode;

	while(leafIndexReverse.size())
		leafIndexReverse.pop_back();

	for(int i=0;i<nNodes;i++)
		if(nodePts[i]->leaf){
			leafIndexReverse.push_back(i);
			leafIndex[i]=leafCount;
			leafCount++;
		}
	nLeaf=leafCount;

	for(int i=0;i<data.size(); i++){
		instance=data[i];
		curNode=nodePts[0]; //curNode points to the root of tree					
		while(curNode->stopP!=1 && !curNode->leaf){
			f=curNode->feature;
			if (instance->features[f]<=curNode->value)
				curNode=curNode->child[BudgetNode::YES];
			else
				curNode=curNode->child[BudgetNode::NO];
		}		
		leafPerExample[i]=leafIndex[curNode->ID];
	}
}

void BudgetTree::getLeafPred(vector<double>& predTmp){
	for(int i=0;i<leafPerExample.size();i++)
		predTmp[i]=nodePts[leafIndexReverse[leafPerExample[i]]]->pred_double;
}

BudgetTree::~BudgetTree(){
	for(int i=0;i<nNodes;i++)
		delete nodePts[i];
}
