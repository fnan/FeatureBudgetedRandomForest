#include "BudgetTree.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

/*constructor*/
BudgetTree::BudgetTree(int nNodes, int numberOfClasses){
	this->nNodes=nNodes;
	this->numberOfClasses=numberOfClasses;
	this->nodePts.resize(nNodes,NULL);
}

/*reset the based on the number of nodes in the tree*/
void BudgetTree::setNNodes(int nNodes){
	this->nNodes=nNodes;
	for(int i=0;i<nNodes;i++){
		this->nodePts.push_back(NULL);
	}
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
	parNode->pred=atoi(strtok(NULL, " \n"));
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

void BudgetTree::writeTree(const BudgetNode* curNode, ofstream& outfile){
	if(curNode==0) return;
	int i;
	int YesID=-1, NoID=-1;
	if(!curNode->leaf){
		YesID=curNode->child[0]->ID;
		NoID=curNode->child[1]->ID;
	}
	//each line: NodeID parID YESID NOID featureID splitValue isLeaf stopP pred [c_leaf] [oob_c_leaf]
	outfile<< curNode->ID << " "<< curNode->parID<<" "<<YesID << " " << NoID << " " << curNode->feature << " " << curNode->value << " " << curNode->leaf << " " << curNode->pred <<" ";
	for(i=0;i<curNode->c_leaf.size();i++)
		outfile<<curNode->c_leaf[i]<<" ";
	for(i=0;i<curNode->oob_c_leaf.size();i++)
		outfile<<curNode->oob_c_leaf[i]<<" ";
	outfile<<endl;
	writeTree(curNode->child[0], outfile);
	writeTree(curNode->child[1],outfile);
}
void BudgetTree::buildLearn(const data_t& train, const vec_data_t& test,args_t& args, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double),std::default_random_engine& gen){
	int kfeatures = args.kfeatures;
	int num_test = test.size();
	int i, ID=0;
	data_t sample;
	std::uniform_int_distribution<int> dist(0,train.size()-1);
	for (i=0; i < train.size(); i++)
		sample.push_back(train[dist(gen)]);

	////quick check for sample randomness:
	//int indexSum=0;
	//for (i=0;i<sample.size();i++)
	//	indexSum+=sample[i]->idx;
	//cout<<  "random sampling indexSum= " <<indexSum<<endl;
	nodePts[0] = new BudgetNode(sample, args, 1, kfeatures, impurityHandle, &ID, -1);
	nNodes=ID;
	updateNodePts();
	if(args.oob>=1){ //include oob samples
		oobUpdate(train,sample); //update leaf distribution using oob samples
		ValUpdate(test[0]); //update leaf distribution using validation samples
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
	vector<int> indx(n,-1);
	double v;
	for(i=0;i<n;i++){
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

BudgetTree::~BudgetTree(){
	for(int i=0;i<nNodes;i++)
		delete nodePts[i];
}