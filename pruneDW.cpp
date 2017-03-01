#include "BudgetForest.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// functor for getting sum of previous result and square of current element
template<typename T>
struct square
{
    T operator()(const T& Left, const T& Right) const
    {   
        return (Left + Right*Right);
    }
};

/*prune the forest using primal-dual sub-gradient ascend given tradeoff parameter lambda*/
void BudgetForest::pruneGA(const args_t& myargs, const data_t& val,  char* writeLPFileName){
	int i,j,k,f,h;
	double v;
	BudgetNode* curNode;
	tupleW* instance;
	int valSize=val.size();
	double lambda=myargs.prune;
	int treesPerProc=nTrees/myargs.processors;

	ofstream outfile;

	/*****compute for each tree and each node a list of example IDs that first encounters the feature at the node*****/
	/*****compute the total number of leaves (including w^t_{k,i} and z_h) under each internal node******/

	vector<vector<vector<int>>> treeNodeExamples;
	vector<vector<int>> treeNodeNumLeaves;

	for(i=0;i<nTrees;i++){
		vector<vector<int>> nodeExamples(nNodes[i],vector<int>());
		vector<int> nodeNumLeaves(nNodes[i],0);
		treeNodeExamples.push_back(nodeExamples);
		treeNodeNumLeaves.push_back(nodeNumLeaves);
	}

	pair<int,int> FeatNode;
	for(i=0;i<val.size();i++){ //iterate through all examples
		for(j=0;j<nTrees;j++){
			vector<pair<int,int>> FeatNodesList;
			//vector<int> TrFeatureListTmp;
			curNode=treePts[j]->nodePts[0]; //curNode points to the root of tree
			instance=val[i];
			while(!curNode->leaf){
				f=curNode->feature;
				k=0;
				for(h=0;h<FeatNodesList.size();h++){
					if(FeatNodesList[h].first==f)
						k=1;
				}
				if(k==0){//feature f has not been added for example i on tree j
					FeatNode=make_pair(f,curNode->ID);
					FeatNodesList.push_back(FeatNode);
					treeNodeExamples[j][curNode->ID].push_back(i);
				}

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
		}
	}
	int parentID;
	for(i=0;i<nTrees;i++){
		for(j=nNodes[i]-1;j>0;j--){
			if(treePts[i]->nodePts[j]->leaf)
				treeNodeNumLeaves[i][j]=1;
			else
				treeNodeNumLeaves[i][j]+=treeNodeExamples[i][j].size();

			parentID=treePts[i]->nodePts[j]->parID;
			treeNodeNumLeaves[i][parentID]+=treeNodeNumLeaves[i][j];
		}
		treeNodeNumLeaves[i][0]+=treeNodeExamples[i][0].size();
	}

	/*count the number of w^t_{k,i}*/

	vector<int> numWtki(nTrees,0);
	vector<int> cumNumWtki(nTrees,0);
	for(i=0;i<nTrees-1;i++){
		for(j=0;j<nNodes[i];j++)
			numWtki[i]+=treeNodeExamples[i][j].size();
		cumNumWtki[i+1]=cumNumWtki[i]+numWtki[i];
	}	
	for(j=0;j<nNodes[i];j++)
		numWtki[i]+=treeNodeExamples[i][j].size();

	/*****compute the number of missclassifications at each node.*******************/
	
	vector<vector<double>> missClass;
	int inbTotal=0, oobTotal=0, sumTmpInb=0, sumTmpOob=0;
	for(i=0;i<nTrees;i++){
		inbTotal=0;
		oobTotal=0;
		vector<double> missVec(treePts[i]->nNodes,0.0);
		for(k=0;k<numberOfClasses;k++){
			inbTotal += treePts[i]->nodePts[0]->c_leaf[k];
			oobTotal += treePts[i]->nodePts[0]->oob_c_leaf[k];
		}
		for(j=0;j<treePts[i]->nNodes;j++){
			sumTmpInb=0;
			sumTmpOob=0;
			for(k=0;k<numberOfClasses;k++){
				sumTmpInb += treePts[i]->nodePts[j]->c_leaf[k];
				sumTmpOob += treePts[i]->nodePts[j]->oob_c_leaf[k];
			}
			if(myargs.oob==0){ //use in-bag samples only to estimate error probability
				missVec[j]=(sumTmpInb-treePts[i]->nodePts[j]->c_leaf[treePts[i]->nodePts[j]->pred])/(double)inbTotal;			
			}
			else if(myargs.oob==1){ //use oob samples only to estimate error probability
				missVec[j]=(sumTmpOob-treePts[i]->nodePts[j]->oob_c_leaf[treePts[i]->nodePts[j]->pred])/(double)oobTotal;
			}
			else{ //use both in-bag and oob samples to estimate error probability
				missVec[j]=(sumTmpInb+sumTmpOob-treePts[i]->nodePts[j]->c_leaf[treePts[i]->nodePts[j]->pred]-treePts[i]->nodePts[j]->oob_c_leaf[treePts[i]->nodePts[j]->pred])/((double)oobTotal+(double)inbTotal);			
			}
		}
		missClass.push_back(missVec);
	}


	/* primal variables w_{k,i}. Linearized for speed*/
	vector<int> wki(myargs.Costs.size()*val.size(),0);
	vector<int> wkiPrimal(myargs.Costs.size()*val.size(),0);
	vector<double> betaKi(myargs.features*val.size(),0.0);

	/* dual variables */
	int numWtkiTotal=0;
	for(i=0;i<nTrees;i++)
		numWtkiTotal+=numWtki[i];
	double * betaTki = (double*)malloc(numWtkiTotal*sizeof(double));
	if(betaTki==NULL){
		fprintf (stderr, "Failed to allocate betaTki.\n");
		return;
	}
	for(i=0;i<numWtkiTotal;i++)
		betaTki[i]=0.0;

	vector<double> betaGrad(numWtkiTotal,0.0);

	/*****************build network problems for each tree ***************************/
	vector<CPXENVptr> envs(myargs.processors,NULL);
	vector<CPXNETptr> networks(nTrees,NULL);
	int       status;
	int rowCounter=0,colCounter=0,counterBeta=0;
	vector<int> nnodes(nTrees,0);
	vector<int> narcs(nTrees,0);
	vector<int>      solstat(nTrees,0);
	vector<double>   objval(nTrees,0);

	int stop=0,iter=0, maxIter=50,m=myargs.learning_rate_base;
	double eta, delta=1; //step size. What's a good value???
	double epsilon=1e-6; //stop threshold
	vector<double> lambdaCostV(myargs.Costs);
	double betaTmp, muTmp,primalObj,dualObj,dualityGap, gradNorm, dualBest=1e8, dualObjOld=0.0;
	vector<double> dualityGapHist(10,1.0);

	/* allocate memory for solution data */
	for(i=0;i<nTrees;i++){
		nnodes[i]=treeNodeNumLeaves[i][0]+1;
		narcs[i]=numWtki[i]+nNodes[i];
	}

	double** x=new double*[nTrees];
	double** xBest = new double*[nTrees];

	int ** objBetaInd=new int*[nTrees];

	for(i=0;i<nTrees;i++){
		x[i] = new double[narcs[i]];
		xBest[i] = new double[narcs[i]];

		objBetaInd[i]   = new int[numWtki[i]];
		if ( x[i]     == NULL ||
			xBest[i]  == NULL ||
			objBetaInd[i]   == NULL) {
			fprintf (stderr, "Failed to allocate x or objBetaInd.\n");
			free(betaTki);
			return;
		}
		for(j=0;j<narcs[i];j++){
			x[i][j]=0.0;
			xBest[i][j]=0.0;
		}
		for(j=0;j<numWtki[i];j++){
			objBetaInd[i][j]=0;
		}
	}

	/* Initialize the CPLEX environments, one per processor */
	for(i=0;i<myargs.processors;i++){
		envs[i] = CPXopenCPLEX (&status);

		/* If an error occurs, the status value indicates the reason for
			failure.  A call to CPXgeterrorstring will produce the text of
			the error message.  Note that CPXopenCPLEX produces no
			output, so the only way to see the cause of the error is to use
			CPXgeterrorstring.  For other CPLEX routines, the errors will
			be seen if the CPXPARAM_ScreenOutput indicator is set to CPX_ON.  */

		if ( envs[i] == NULL ) {
			char  errmsg[CPXMESSAGEBUFSIZE];
			fprintf (stderr, "Could not open CPLEX environment.\n");
			CPXgeterrorstring (envs[i], status, errmsg);
			fprintf (stderr, "%s", errmsg);
			free(betaTki);
			for(i=0;i<nTrees;i++){
				delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
			}
			delete [] x; delete[] xBest; delete [] objBetaInd;
			return;
		}

		///*set tolerence*/
		//status = CPXsetlongparam(envs[i],CPX_PARAM_NETEPOPT,1e-11);
		//if( status ){
		//	fprintf (stderr, 
		//			"Failure to set network optimality tolerence, error %d.\n", status);
		//	free(betaTki);
		//	for(i=0;i<nTrees;i++){
		//		delete[] x[i]; delete [] objBetaInd[i];
		//	}
		//	delete [] x; delete [] objBetaInd;
		//	return;		
		//}
		/* Turn on output to the screen */

		//status = CPXsetintparam (envs[i], CPXPARAM_ScreenOutput, CPX_ON);
		//if ( status ) {
		//	fprintf (stderr, 
		//			"Failure to turn on screen indicator, error %d.\n", status);
		//	free(betaTki);
		//	for(i=0;i<nTrees;i++){
		//		delete[] x[i]; delete [] objBetaInd[i];
		//	}
		//	delete [] x; delete [] objBetaInd;
		//	return;
		//}
	}

	/* Create the subproblem for each tree. */
	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			double *supply=(double *) malloc (nnodes[i] * sizeof (double));
			int *tail=(int *) malloc(narcs[i] * sizeof(int));
			int *head=(int*)malloc(narcs[i]*sizeof(int));
			double *obj=(double*)malloc(narcs[i]*sizeof(double));

			networks[i] = CPXNETcreateprob (envs[p], &status, "networkTree");

			/* A returned pointer of NULL may mean that not enough memory
				was available or there was some other problem.  In the case of 
				failure, an error message will have been written to the error 
				channel from inside CPLEX.  In this example, the setting of
				the parameter CPXPARAM_ScreenOutput causes the error message to
				appear on stdout.  */

			if ( networks[i] == NULL ) {
				fprintf (stderr, "Failed to create network object.\n");
				break;
			}

			/* Fill in the data for the problem.  Note that since the space for
				the data already exists in local variables, we pass the arrays
				directly to the routine to fill in the data structures.  */

	//		status = buildNetwork(env, networks[i],treeNodeNumLeaves[i],treeNodeExamples[i],numWtki[i],missClass[i]);

			for(j=0;j<nnodes[i];j++)
				supply[j]=0.0;
			supply[0]=1.0;
			supply[nnodes[i]-1]=-1.0;

			rowCounter=0; colCounter=0;
			for(j=0;j<nNodes[i];j++){
				obj[colCounter]=missClass[i][j]/nTrees;
				tail[colCounter]=rowCounter;
				head[colCounter]=rowCounter+treeNodeNumLeaves[i][j];
				if(treeNodeNumLeaves[i][j]==1)
					rowCounter++;
				colCounter++;
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					obj[colCounter]=0.0;
					tail[colCounter]=rowCounter;
					rowCounter++;
					head[colCounter]=rowCounter;
					colCounter++;
				}
			}
			/* Set optimization sense */

			status = CPXNETchgobjsen (envs[p], networks[i], CPX_MIN);
			if ( status ) break;

			/* Add nodes to network along with their supply values,
				but without any names. */

			status = CPXNETaddnodes (envs[p], networks[i], nnodes[i], supply, NULL);
			if ( status ) break;

			/* Add arcs to network along with their objective values and
				bounds, but without any names. */

			status = CPXNETaddarcs (envs[p], networks[i], narcs[i], tail, head, NULL, NULL, obj, NULL);
			if ( status ) break;

			free(supply); free(tail); free(head); free(obj);
		}
	}
	if ( status ) {
		fprintf (stderr, "Failed to build network problem.\n");
		free(betaTki);
		for(i=0;i<nTrees;i++){
			delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
		}
		delete [] x; delete[] xBest; delete [] objBetaInd;
		return;
	}
	

	/* build index of dual variables in terms of arc indices, will be used to update network problem objectives */
	for(i=0;i<nTrees;i++){
		colCounter=0;
		counterBeta=0;
		for(j=0;j<nNodes[i];j++){
			colCounter++;
			for(k=0;k<treeNodeExamples[i][j].size();k++){
				objBetaInd[i][counterBeta]=colCounter;
				counterBeta++; colCounter++;
			}
		}
	}

	/* iterate between master and subproblems until convergence*/

	for(i=0;i<myargs.features;i++)
		lambdaCostV[i]=myargs.Costs[i]*lambda/valSize;

	/*set initial basis for the network problem*/
	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			int *arc_stat= new int[narcs[i]];
			int *node_stat= new int[nnodes[i]];
			colCounter=0;
			for(j=0;j<nNodes[i];j++){
				if(treePts[i]->nodePts[j]->leaf==1)
					arc_stat[colCounter]=CPX_BASIC;
				else
					arc_stat[colCounter]=CPX_AT_LOWER;
				colCounter++;
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					arc_stat[colCounter]=CPX_BASIC;
					colCounter++;
				}
			}
			node_stat[0]=CPX_BASIC;
			for(j=1;j<nnodes[i];j++){
				node_stat[j]=CPX_AT_LOWER;
			}
			status = CPXNETcopybase (envs[p], networks[i], arc_stat, node_stat);
			if ( status ) {
				fprintf (stderr, "Failed to set initial solution.\n");
				free(betaTki);
				for(i=0;i<nTrees;i++){
					delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
				}
				delete [] x; delete[] xBest; delete [] objBetaInd;
				return;
			}
			delete [] arc_stat; delete [] node_stat;
		}
	}
	while(!stop && iter<maxIter){
		/* solve the subproblems */

		/* \sum_t betaTki to solve for wki */
		std::fill(wki.begin(),wki.end(),0);
		std::fill(wkiPrimal.begin(),wkiPrimal.end(),0);
		std::fill(betaKi.begin(),betaKi.end(),0.0);

		counterBeta=0;
		for(i=0;i<nTrees;i++){
			for(j=0;j<nNodes[i];j++){
				f=treePts[i]->nodePts[j]->feature;
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					betaKi[valSize*f+treeNodeExamples[i][j][k]]+=betaTki[counterBeta];
					counterBeta++;
				}
			}
		}

		primalObj=0.0;
		dualObj=0.0;
		for(i=1;i<myargs.features;i++){
			for(j=0;j<valSize;j++){
				muTmp=lambdaCostV[i]-betaKi[valSize*i+j];
				if(muTmp>=0)
					wki[valSize*i+j]=0.0;
				else{
					wki[valSize*i+j]=1.0;
					dualObj += muTmp;				
				}
			}
		}
		pruningInp pruneInp;
		pruneInp.numTrees = treesPerProc;

		/* Optimize the subproblems and obtain solution. */
		thread** threads = new thread*[myargs.processors];
		for (i = 0; i < myargs.processors; i++){
			pruneInp.start = i*treesPerProc;
			threads[i] = new thread(&BudgetForest::solveNetworkPerProc, this, ref(envs), ref(networks),  pruneInp, ref(status));
		}
		for (i=0;i<myargs.processors;i++){
			threads[i]->join();
			delete threads[i];
		}

		for(int p=0;p<myargs.processors;p++){
			for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){

				//status = CPXNETprimopt (env, networks[i]);
				//if ( status ) {
				//	fprintf (stderr, "Failed to optimize network.\n");
				//	free(betaTki);
				//	for(i=0;i<nTrees;i++){
				//		delete[] x[i]; delete [] objBetaInd[i];
				//	}
				//	delete [] x; delete [] objBetaInd;
				//	return;
				//}
				/*debug output: 
				for(int ii=0;ii<nTrees;ii++)
					for(j=0;j<numWtki[ii];j++)
						cout<<"objBetaInd["<<ii<<"]["<<j<<"]="<<objBetaInd[ii][j]<<endl;
				*/
	//			status = CPXNETsolution (env, networks[i], &solstat[i], &objval[i], x[i], pi[i], slack[i], dj[i]);
	//for(int ii=0;ii<narcs[i];ii++)
	//	cout<<x[i][ii]<<endl;

				status = CPXNETgetx (envs[p], networks[i], x[i], 0, narcs[i]-1);
				if ( status ) {
					fprintf (stderr, "Failed to obtain solution.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}
				/*debug output: 
				cout<<"after obtain solution"<<endl;
				for(int ii=0;ii<narcs[i];ii++)
					cout<<x[i][ii]<<endl;
				for(int ii=0;ii<nTrees;ii++)
					for(j=0;j<numWtki[ii];j++)
						cout<<"objBetaInd["<<ii<<"]["<<j<<"]="<<objBetaInd[ii][j]<<endl;
				*/
				status = CPXNETgetobjval (envs[p], networks[i], &objval[i]);
				if ( status ) {
					fprintf (stderr, "Failed to obtain objective.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}

				/*extract the flow solution while computing the primal objective */
				colCounter=0;
				counterBeta=cumNumWtki[i];
				for(j=0;j<nNodes[i];j++){
					primalObj+=x[i][colCounter]*missClass[i][j]/nTrees;
					colCounter++;
					f=treePts[i]->nodePts[j]->feature;
					for(k=0;k<treeNodeExamples[i][j].size();k++){
						if(x[i][colCounter]>0.9999)
							wkiPrimal[f*valSize+treeNodeExamples[i][j][k]]=1;
						betaGrad[counterBeta]=x[i][colCounter]-wki[valSize*f+treeNodeExamples[i][j][k]];
						colCounter++; counterBeta++;
					}
				}
			}
		}
		for(i=1;i<lambdaCostV.size();i++){
			primalObj += std::accumulate(wkiPrimal.begin()+i*valSize, wkiPrimal.begin()+(i+1)*valSize,0)*lambdaCostV[i];			
		}
		for(i=0;i<nTrees;i++)
			dualObj+=objval[i];
		/*check for convergence based on duality gap*/
		dualityGap=primalObj-dualObj;
		dualityGapHist[iter%10]=dualityGap;

		iter++;

		/* projected subgradient ascend step for the master problem */
		/* step size rule using Bertsekas p.623 , dualBest keeps track of the max dual objective so far */
		gradNorm= std::accumulate( betaGrad.begin(), betaGrad.end(), 0, square<double>() );

		//if(dualObj>dualObjOld)
		//	delta*=2;
		//else
		//	delta*=0.5;

		//dualObjOld=dualObj;

		//if(dualBest<dualObj)
		//	dualBest=dualObj;

		//if(gradNorm<epsilon){
		//	eta=1;
		//}else{
		//	eta=(dualBest+delta-dualObj)/gradNorm;
		//}

		if(dualBest>primalObj){
			dualBest=primalObj;
			for(i=0;i<nTrees;i++){ //copy the best primal solution
				std::copy(x[i],x[i]+narcs[i],xBest[i]);
			}
		}
		if(gradNorm<epsilon){
			eta=1;
		}else{
			eta=(1+m)*(dualBest-dualObj)/(gradNorm*(iter+m));
		}

		cout<< "primal-dual iteration: "<< iter << " | " << primalObj << " | " << dualObj<< " | " << dualityGap <<endl;
		//cout << "primal-dual iteration: " << iter << " | " << primalObj << " | " << dualObj << " | " << dualityGap << " | " << std::accumulate(betaTki, betaTki + numWtkiTotal, 0.0) << endl;
		//		if(dualityGap<epsilon || (dualityGap<0.001 && std::accumulate(dualityGapHist.begin(), dualityGapHist.end(),0.0)/10-dualityGap<dualityGap*0.00000001)){
		if(dualityGap<epsilon || (iter>1000 && std::accumulate(dualityGapHist.begin(), dualityGapHist.end(),0.0)/10-dualityGap<dualityGap*0.00000001)){
			cout<< "Termination: primal-dual done!" <<endl;
			stop=1;
			continue;
		}

		for(i=0;i<numWtkiTotal;i++){
			betaTmp=betaTki[i]+eta*betaGrad[i];
			if(betaTmp<0)
				betaTki[i]=0.0;
			else
				betaTki[i]=betaTmp;
		}

		/*update the subproblem objectives based on solution of master */
		counterBeta=0;
		for(int p=0;p<myargs.processors;p++){
			for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
				status = CPXNETchgobj (envs[p], networks[i], numWtki[i], objBetaInd[i], betaTki+cumNumWtki[i]);
				if ( status ) {
					fprintf (stderr, "Failed to update network objective.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}
			}
		}
	}//iterate between primal and dual

			/* update the stopP for all nodes */

	for(i=0;i<nTrees;i++){
		colCounter=0;
		for(j=0;j<nNodes[i];j++){
			treePts[i]->nodePts[j]->stopP=xBest[i][colCounter];
			colCounter+=treeNodeExamples[i][j].size()+1;
		}
		for(j=1;j<nNodes[i];j++){
			parentID=treePts[i]->nodePts[j]->parID;
			if(treePts[i]->nodePts[parentID]->stopP>0.99999)
				treePts[i]->nodePts[j]->stopP=1.0;
		}
	}	
	printf("best primal objective: %.17e\n",dualBest);
	if(stop && dualityGap<epsilon){
		cout<< " primal-dual solution found! Saving pruning solutions" << endl;
	}
	else{
		cout<< "primal-dual failed!"<<endl;
	}

	/*free memory*/
	if (betaTki != NULL){
		free(betaTki);
		betaTki=NULL;
	}
	for(i=0;i<nTrees;i++){
		if(x[i]!=NULL)
			delete [] x[i];
		if(xBest[i]!=NULL)
			delete [] xBest[i];
		//free(dj[i]);
		//free(pi[i]);
		//free(slack[i]);
		if(objBetaInd[i]!=NULL)
			delete [] objBetaInd[i];
	}
	delete [] x; delete [] xBest; delete [] objBetaInd;
	/* Free up the problem as allocated by CPXNETcreateprob, if necessary */

	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			if ( networks[i] != NULL ) {
				status = CPXNETfreeprob (envs[p], &networks[i]);
				if ( status ) {
					fprintf (stderr, "CPXNETfreeprob failed, error code %d.\n", status);
				}
			}
		}
	}
	/* Free up the CPLEX environment, if necessary */
	for(int p=0; p<myargs.processors;p++){
		if ( envs[p] != NULL ) {
			status = CPXcloseCPLEX (&envs[p]);

			/* Note that CPXcloseCPLEX produces no output,
				so the only way to see the cause of the error is to use
				CPXgeterrorstring.  For other CPLEX routines, the errors will
				be seen if the CPXPARAM_ScreenOutput indicator is set to CPX_ON. */

			if ( status ) {
			char  errmsg[CPXMESSAGEBUFSIZE];
				fprintf (stderr, "Could not close CPLEX environment.\n");
				CPXgeterrorstring (envs[p], status, errmsg);
				fprintf (stderr, "%s", errmsg);
			}
		}
	}
}
void BudgetForest::solveNetworkPerProc(vector<CPXENVptr>& envs, vector<CPXNETptr>& networks, const pruningInp pruneInp, int& status){
	for(int i=pruneInp.start;i<pruneInp.start+pruneInp.numTrees;i++){
//		cout<<"solving network:"<<i<<"| start="<<pruneInp.start<< " | numTreesPerProc=" << pruneInp.numTrees <<endl;
		status = CPXNETprimopt(envs[pruneInp.start / pruneInp.numTrees], networks[i]);
			if ( status ) {
				fprintf (stderr, "Failed to optimize network.\n");
				return;
			}	
	}
}

/*prune the forest using primal-dual sub-gradient ascend given tradeoff parameter lambda*/
void BudgetForest::pruneGAgrp(const args_t& myargs, const data_t& val,  char* writeLPFileName){
	int i,j,k,f,h;
	double v;
	BudgetNode* curNode;
	tupleW* instance;
	int valSize=val.size();
	double lambda=myargs.prune;
	int treesPerProc=nTrees/myargs.processors;

	ofstream outfile;

	/*****compute for each tree and each node a list of example IDs that first encounters the feature at the node*****/
	/*****compute the total number of leaves (including w^t_{k,i} and z_h) under each internal node******/

	vector<vector<vector<int>>> treeNodeExamples;
	vector<vector<int>> treeNodeNumLeaves;

	for(i=0;i<nTrees;i++){
		vector<vector<int>> nodeExamples(nNodes[i],vector<int>());
		vector<int> nodeNumLeaves(nNodes[i],0);
		treeNodeExamples.push_back(nodeExamples);
		treeNodeNumLeaves.push_back(nodeNumLeaves);
	}

	pair<int,int> SensorNode;
	for(i=0;i<val.size();i++){ //iterate through all examples
		for(j=0;j<nTrees;j++){
			vector<pair<int,int>> SensorNodesList;
			//vector<int> TrFeatureListTmp;
			curNode=treePts[j]->nodePts[0]; //curNode points to the root of tree
			instance=val[i];
			while(!curNode->leaf){
				f=curNode->feature;
				k=0;
				for(h=0;h<SensorNodesList.size();h++){
					if(SensorNodesList[h].first==myargs.Costgroup[f])
						k=1;
				}
				if(k==0){//feature f has not been added for example i on tree j
					SensorNode=make_pair(myargs.Costgroup[f],curNode->ID);
					SensorNodesList.push_back(SensorNode);
					treeNodeExamples[j][curNode->ID].push_back(i);
				}

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
		}
	}
	int parentID;
	for(i=0;i<nTrees;i++){
		for(j=nNodes[i]-1;j>0;j--){
			if(treePts[i]->nodePts[j]->leaf)
				treeNodeNumLeaves[i][j]=1;
			else
				treeNodeNumLeaves[i][j]+=treeNodeExamples[i][j].size();

			parentID=treePts[i]->nodePts[j]->parID;
			treeNodeNumLeaves[i][parentID]+=treeNodeNumLeaves[i][j];
		}
		treeNodeNumLeaves[i][0]+=treeNodeExamples[i][0].size();
	}

	/*count the number of w^t_{k,i}*/

	vector<int> numWtki(nTrees,0);
	vector<int> cumNumWtki(nTrees,0);
	for(i=0;i<nTrees-1;i++){
		for(j=0;j<nNodes[i];j++)
			numWtki[i]+=treeNodeExamples[i][j].size();
		cumNumWtki[i+1]=cumNumWtki[i]+numWtki[i];
	}	
	for(j=0;j<nNodes[i];j++)
		numWtki[i]+=treeNodeExamples[i][j].size();

	/*****compute the number of missclassifications at each node.*******************/
	
	vector<vector<double>> missClass;
	int inbTotal=0, oobTotal=0, sumTmpInb=0, sumTmpOob=0;
	for(i=0;i<nTrees;i++){
		inbTotal=0;
		oobTotal=0;
		vector<double> missVec(treePts[i]->nNodes,0.0);
		for(k=0;k<numberOfClasses;k++){
			inbTotal += treePts[i]->nodePts[0]->c_leaf[k];
			oobTotal += treePts[i]->nodePts[0]->oob_c_leaf[k];
		}
		for(j=0;j<treePts[i]->nNodes;j++){
			sumTmpInb=0;
			sumTmpOob=0;
			for(k=0;k<numberOfClasses;k++){
				sumTmpInb += treePts[i]->nodePts[j]->c_leaf[k];
				sumTmpOob += treePts[i]->nodePts[j]->oob_c_leaf[k];
			}
			if(myargs.oob==0){ //use in-bag samples only to estimate error probability
				missVec[j]=(sumTmpInb-treePts[i]->nodePts[j]->c_leaf[treePts[i]->nodePts[j]->pred])/(double)inbTotal;			
			}
			else if(myargs.oob==1){ //use oob samples only to estimate error probability
				missVec[j]=(sumTmpOob-treePts[i]->nodePts[j]->oob_c_leaf[treePts[i]->nodePts[j]->pred])/(double)oobTotal;
			}
			else{ //use both in-bag and oob samples to estimate error probability
				missVec[j]=(sumTmpInb+sumTmpOob-treePts[i]->nodePts[j]->c_leaf[treePts[i]->nodePts[j]->pred]-treePts[i]->nodePts[j]->oob_c_leaf[treePts[i]->nodePts[j]->pred])/((double)oobTotal+(double)inbTotal);			
			}
		}
		missClass.push_back(missVec);
	}

	
	/* primal variables w_{k,i}. Linearized for speed*/
	vector<int> wki(myargs.sensors*val.size(),0);
	vector<int> wkiPrimal(myargs.sensors*val.size(),0);
	vector<double> betaKi(myargs.sensors*val.size(),0.0);

	/* dual variables */
	int numWtkiTotal=0;
	for(i=0;i<nTrees;i++)
		numWtkiTotal+=numWtki[i];
	double * betaTki = (double*)malloc(numWtkiTotal*sizeof(double));
	if(betaTki==NULL){
		fprintf (stderr, "Failed to allocate betaTki.\n");
		return;
	}
	for(i=0;i<numWtkiTotal;i++)
		betaTki[i]=0.0;

	vector<double> betaGrad(numWtkiTotal,0.0);

	/*****************build network problems for each tree ***************************/
	vector<CPXENVptr> envs(myargs.processors,NULL);
	vector<CPXNETptr> networks(nTrees,NULL);
	int       status;
	int rowCounter=0,colCounter=0,counterBeta=0;
	vector<int> nnodes(nTrees,0);
	vector<int> narcs(nTrees,0);
	vector<int>      solstat(nTrees,0);
	vector<double>   objval(nTrees,0);

	int stop=0,iter=0, maxIter=100,m=myargs.learning_rate_base;
	double eta, delta=1; //step size. What's a good value???
	double epsilon=1e-6; //stop threshold
	vector<double> lambdaCostV(myargs.CostSensors);
	double betaTmp, muTmp,primalObj,dualObj,dualityGap, gradNorm, dualBest=1e8, dualObjOld=0.0;
	vector<double> dualityGapHist(10,1.0);

	/* allocate memory for solution data */
	for(i=0;i<nTrees;i++){
		nnodes[i]=treeNodeNumLeaves[i][0]+1;
		narcs[i]=numWtki[i]+nNodes[i];
	}

	double** x=new double*[nTrees];
	double** xBest = new double*[nTrees];

	int ** objBetaInd=new int*[nTrees];

	for(i=0;i<nTrees;i++){
		x[i] = new double[narcs[i]];
		xBest[i] = new double[narcs[i]];

		objBetaInd[i]   = new int[numWtki[i]];
		if ( x[i]     == NULL ||
			xBest[i]  == NULL ||
			objBetaInd[i]   == NULL) {
			fprintf (stderr, "Failed to allocate x or objBetaInd.\n");
			free(betaTki);
			return;
		}
		for(j=0;j<narcs[i];j++){
			x[i][j]=0.0;
			xBest[i][j]=0.0;
		}
		for(j=0;j<numWtki[i];j++){
			objBetaInd[i][j]=0;
		}
	}

	/* Initialize the CPLEX environments, one per processor */
	for(i=0;i<myargs.processors;i++){
		envs[i] = CPXopenCPLEX (&status);

		/* If an error occurs, the status value indicates the reason for
			failure.  A call to CPXgeterrorstring will produce the text of
			the error message.  Note that CPXopenCPLEX produces no
			output, so the only way to see the cause of the error is to use
			CPXgeterrorstring.  For other CPLEX routines, the errors will
			be seen if the CPXPARAM_ScreenOutput indicator is set to CPX_ON.  */

		if ( envs[i] == NULL ) {
			char  errmsg[CPXMESSAGEBUFSIZE];
			fprintf (stderr, "Could not open CPLEX environment.\n");
			CPXgeterrorstring (envs[i], status, errmsg);
			fprintf (stderr, "%s", errmsg);
			free(betaTki);
			for(i=0;i<nTrees;i++){
				delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
			}
			delete [] x; delete[] xBest; delete [] objBetaInd;
			return;
		}

		///*set tolerence*/
		//status = CPXsetlongparam(envs[i],CPX_PARAM_NETEPOPT,1e-11);
		//if( status ){
		//	fprintf (stderr, 
		//			"Failure to set network optimality tolerence, error %d.\n", status);
		//	free(betaTki);
		//	for(i=0;i<nTrees;i++){
		//		delete[] x[i]; delete [] objBetaInd[i];
		//	}
		//	delete [] x; delete [] objBetaInd;
		//	return;		
		//}
		/* Turn on output to the screen */

		//status = CPXsetintparam (envs[i], CPXPARAM_ScreenOutput, CPX_ON);
		//if ( status ) {
		//	fprintf (stderr, 
		//			"Failure to turn on screen indicator, error %d.\n", status);
		//	free(betaTki);
		//	for(i=0;i<nTrees;i++){
		//		delete[] x[i]; delete [] objBetaInd[i];
		//	}
		//	delete [] x; delete [] objBetaInd;
		//	return;
		//}
	}

	/* Create the subproblem for each tree. */
	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			double *supply=(double *) malloc (nnodes[i] * sizeof (double));
			int *tail=(int *) malloc(narcs[i] * sizeof(int));
			int *head=(int*)malloc(narcs[i]*sizeof(int));
			double *obj=(double*)malloc(narcs[i]*sizeof(double));

			networks[i] = CPXNETcreateprob (envs[p], &status, "networkTree");

			/* A returned pointer of NULL may mean that not enough memory
				was available or there was some other problem.  In the case of 
				failure, an error message will have been written to the error 
				channel from inside CPLEX.  In this example, the setting of
				the parameter CPXPARAM_ScreenOutput causes the error message to
				appear on stdout.  */

			if ( networks[i] == NULL ) {
				fprintf (stderr, "Failed to create network object.\n");
				break;
			}

			/* Fill in the data for the problem.  Note that since the space for
				the data already exists in local variables, we pass the arrays
				directly to the routine to fill in the data structures.  */

	//		status = buildNetwork(env, networks[i],treeNodeNumLeaves[i],treeNodeExamples[i],numWtki[i],missClass[i]);

			for(j=0;j<nnodes[i];j++)
				supply[j]=0.0;
			supply[0]=1.0;
			supply[nnodes[i]-1]=-1.0;

			rowCounter=0; colCounter=0;
			for(j=0;j<nNodes[i];j++){
				obj[colCounter]=missClass[i][j]/nTrees;
				tail[colCounter]=rowCounter;
				head[colCounter]=rowCounter+treeNodeNumLeaves[i][j];
				if(treeNodeNumLeaves[i][j]==1)
					rowCounter++;
				colCounter++;
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					obj[colCounter]=0.0;
					tail[colCounter]=rowCounter;
					rowCounter++;
					head[colCounter]=rowCounter;
					colCounter++;
				}
			}
			/* Set optimization sense */

			status = CPXNETchgobjsen (envs[p], networks[i], CPX_MIN);
			if ( status ) break;

			/* Add nodes to network along with their supply values,
				but without any names. */

			status = CPXNETaddnodes (envs[p], networks[i], nnodes[i], supply, NULL);
			if ( status ) break;

			/* Add arcs to network along with their objective values and
				bounds, but without any names. */

			status = CPXNETaddarcs (envs[p], networks[i], narcs[i], tail, head, NULL, NULL, obj, NULL);
			if ( status ) break;

			free(supply); free(tail); free(head); free(obj);
		}
	}
	if ( status ) {
		fprintf (stderr, "Failed to build network problem.\n");
		free(betaTki);
		for(i=0;i<nTrees;i++){
			delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
		}
		delete [] x; delete[] xBest; delete [] objBetaInd;
		return;
	}
	

	/* build index of dual variables in terms of arc indices, will be used to update network problem objectives */
	for(i=0;i<nTrees;i++){
		colCounter=0;
		counterBeta=0;
		for(j=0;j<nNodes[i];j++){
			colCounter++;
			for(k=0;k<treeNodeExamples[i][j].size();k++){
				objBetaInd[i][counterBeta]=colCounter;
				counterBeta++; colCounter++;
			}
		}
	}

	/* iterate between master and subproblems until convergence*/

	for(i=0;i<myargs.sensors;i++)
		lambdaCostV[i]=myargs.CostSensors[i]*lambda/valSize;

	/*set initial basis for the network problem*/
	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			int *arc_stat= new int[narcs[i]];
			int *node_stat= new int[nnodes[i]];
			colCounter=0;
			for(j=0;j<nNodes[i];j++){
				if(treePts[i]->nodePts[j]->leaf==1)
					arc_stat[colCounter]=CPX_BASIC;
				else
					arc_stat[colCounter]=CPX_AT_LOWER;
				colCounter++;
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					arc_stat[colCounter]=CPX_BASIC;
					colCounter++;
				}
			}
			node_stat[0]=CPX_BASIC;
			for(j=1;j<nnodes[i];j++){
				node_stat[j]=CPX_AT_LOWER;
			}
			status = CPXNETcopybase (envs[p], networks[i], arc_stat, node_stat);
			if ( status ) {
				fprintf (stderr, "Failed to set initial solution.\n");
				free(betaTki);
				for(i=0;i<nTrees;i++){
					delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
				}
				delete [] x; delete[] xBest; delete [] objBetaInd;
				return;
			}
			delete [] arc_stat; delete [] node_stat;
		}
	}
	while(!stop && iter<maxIter){
		/* solve the subproblems */

		/* \sum_t betaTki to solve for wki */
		std::fill(wki.begin(),wki.end(),0);
		std::fill(wkiPrimal.begin(),wkiPrimal.end(),0);
		std::fill(betaKi.begin(),betaKi.end(),0.0);

		counterBeta=0;
		for(i=0;i<nTrees;i++){
			for(j=0;j<nNodes[i];j++){
				f=treePts[i]->nodePts[j]->feature;
				int sensor=myargs.Costgroup[f];
				for(k=0;k<treeNodeExamples[i][j].size();k++){
					betaKi[valSize*sensor+treeNodeExamples[i][j][k]]+=betaTki[counterBeta];
					counterBeta++;
				}
			}
		}

		primalObj=0.0;
		dualObj=0.0;
		for(i=1;i<myargs.sensors;i++){
			for(j=0;j<valSize;j++){
				muTmp=lambdaCostV[i]-betaKi[valSize*i+j];
				if(muTmp>=0)
					wki[valSize*i+j]=0.0;
				else{
					wki[valSize*i+j]=1.0;
					dualObj += muTmp;				
				}
			}
		}

		pruningInp pruneInp;
		pruneInp.numTrees = treesPerProc;

		/* Optimize the subproblems and obtain solution. */
		thread** threads = new thread*[myargs.processors];
		for (i = 0; i < myargs.processors; i++){
			pruneInp.start = i*treesPerProc;
			threads[i] = new thread(&BudgetForest::solveNetworkPerProc, this, ref(envs), ref(networks), pruneInp, ref(status));
		}
		for (i = 0; i<myargs.processors; i++){
			threads[i]->join();
			delete threads[i];
		}

		for(int p=0;p<myargs.processors;p++){
			for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){

				//status = CPXNETprimopt (env, networks[i]);
				//if ( status ) {
				//	fprintf (stderr, "Failed to optimize network.\n");
				//	free(betaTki);
				//	for(i=0;i<nTrees;i++){
				//		delete[] x[i]; delete [] objBetaInd[i];
				//	}
				//	delete [] x; delete [] objBetaInd;
				//	return;
				//}
				/*debug output: 
				for(int ii=0;ii<nTrees;ii++)
					for(j=0;j<numWtki[ii];j++)
						cout<<"objBetaInd["<<ii<<"]["<<j<<"]="<<objBetaInd[ii][j]<<endl;
				*/
	//			status = CPXNETsolution (env, networks[i], &solstat[i], &objval[i], x[i], pi[i], slack[i], dj[i]);
	//for(int ii=0;ii<narcs[i];ii++)
	//	cout<<x[i][ii]<<endl;

				status = CPXNETgetx (envs[p], networks[i], x[i], 0, narcs[i]-1);
				if ( status ) {
					fprintf (stderr, "Failed to obtain solution.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}
				/*debug output: 
				cout<<"after obtain solution"<<endl;
				for(int ii=0;ii<narcs[i];ii++)
					cout<<x[i][ii]<<endl;
				for(int ii=0;ii<nTrees;ii++)
					for(j=0;j<numWtki[ii];j++)
						cout<<"objBetaInd["<<ii<<"]["<<j<<"]="<<objBetaInd[ii][j]<<endl;
				*/
				status = CPXNETgetobjval (envs[p], networks[i], &objval[i]);
				if ( status ) {
					fprintf (stderr, "Failed to obtain objective.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}

				/*extract the flow solution while computing the primal objective */
				colCounter=0;
				counterBeta=cumNumWtki[i];
				for(j=0;j<nNodes[i];j++){
					primalObj+=x[i][colCounter]*missClass[i][j]/nTrees;
					colCounter++;
					f=treePts[i]->nodePts[j]->feature;
					int sensor=myargs.Costgroup[f];
					for(k=0;k<treeNodeExamples[i][j].size();k++){
						if(x[i][colCounter]>0.9999)
							wkiPrimal[sensor*valSize+treeNodeExamples[i][j][k]]=1;
						betaGrad[counterBeta]=x[i][colCounter]-wki[valSize*sensor+treeNodeExamples[i][j][k]];
						colCounter++; counterBeta++;
					}
				}
			}
		}
		for(i=1;i<lambdaCostV.size();i++){
			primalObj += std::accumulate(wkiPrimal.begin()+i*valSize, wkiPrimal.begin()+(i+1)*valSize,0)*lambdaCostV[i];			
		}
		for(i=0;i<nTrees;i++)
			dualObj+=objval[i];
		/*check for convergence based on duality gap*/
		dualityGap=primalObj-dualObj;
		dualityGapHist[iter%10]=dualityGap;

		iter++;

		/* projected subgradient ascend step for the master problem */
		/* step size rule using Bertsekas p.623 , dualBest keeps track of the max dual objective so far */
		gradNorm= std::accumulate( betaGrad.begin(), betaGrad.end(), 0, square<double>() );

		//if(dualObj>dualObjOld)
		//	delta*=2;
		//else
		//	delta*=0.5;

		//dualObjOld=dualObj;

		//if(dualBest<dualObj)
		//	dualBest=dualObj;

		//if(gradNorm<epsilon){
		//	eta=1;
		//}else{
		//	eta=(dualBest+delta-dualObj)/gradNorm;
		//}

		if(dualBest>primalObj){
			dualBest=primalObj;
			for(i=0;i<nTrees;i++){ //copy the best primal solution
				std::copy(x[i],x[i]+narcs[i],xBest[i]);
			}
		}
		if(gradNorm<epsilon){
			eta=1;
		}else{
			eta=(1+m)*(dualBest-dualObj)/(gradNorm*(iter+m));
		}

		cout<< "primal-dual iteration: "<< iter << " | " << primalObj << " | " << dualObj<< " | " << dualityGap <<" | "<<std::accumulate(betaTki,betaTki+numWtkiTotal,0.0) <<endl;
//		if(dualityGap<epsilon || (dualityGap<0.001 && std::accumulate(dualityGapHist.begin(), dualityGapHist.end(),0.0)/10-dualityGap<dualityGap*0.00000001)){
		if(dualityGap<epsilon || (iter>1000 && std::accumulate(dualityGapHist.begin(), dualityGapHist.end(),0.0)/10-dualityGap<dualityGap*0.00000001)){
			cout<< "Termination: primal-dual done!" <<endl;
			stop=1;
			continue;
		}

		for(i=0;i<numWtkiTotal;i++){
			betaTmp=betaTki[i]+eta*betaGrad[i];
			if(betaTmp<0)
				betaTki[i]=0.0;
			else
				betaTki[i]=betaTmp;
		}

		/*update the subproblem objectives based on solution of master */
		counterBeta=0;
		for(int p=0;p<myargs.processors;p++){
			for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
				status = CPXNETchgobj (envs[p], networks[i], numWtki[i], objBetaInd[i], betaTki+cumNumWtki[i]);
				if ( status ) {
					fprintf (stderr, "Failed to update network objective.\n");
					free(betaTki);
					for(i=0;i<nTrees;i++){
						delete[] x[i]; delete[] xBest[i]; delete [] objBetaInd[i];
					}
					delete [] x; delete[] xBest; delete [] objBetaInd;
					return;
				}
			}
		}
	}//iterate between primal and dual

			/* update the stopP for all nodes */

	for(i=0;i<nTrees;i++){
		colCounter=0;
		for(j=0;j<nNodes[i];j++){
			treePts[i]->nodePts[j]->stopP=xBest[i][colCounter];
			colCounter+=treeNodeExamples[i][j].size()+1;
		}
		for(j=1;j<nNodes[i];j++){
			parentID=treePts[i]->nodePts[j]->parID;
			if(treePts[i]->nodePts[parentID]->stopP>0.99999)
				treePts[i]->nodePts[j]->stopP=1.0;
		}
	}	
	printf("best primal objective: %.17e\n",dualBest);
	if(stop && dualityGap<epsilon){
		cout<< " primal-dual solution found! Saving pruning solutions" << endl;
	}
	else{
		cout<< "primal-dual failed!"<<endl;
	}

	/*free memory*/
	if (betaTki != NULL){
		free(betaTki);
		betaTki=NULL;
	}
	for(i=0;i<nTrees;i++){
		if(x[i]!=NULL)
			delete [] x[i];
		if(xBest[i]!=NULL)
			delete [] xBest[i];
		//free(dj[i]);
		//free(pi[i]);
		//free(slack[i]);
		if(objBetaInd[i]!=NULL)
			delete [] objBetaInd[i];
	}
	delete [] x; delete [] xBest; delete [] objBetaInd;
	/* Free up the problem as allocated by CPXNETcreateprob, if necessary */

	for(int p=0;p<myargs.processors;p++){
		for(i=p*treesPerProc;i<(p+1)*treesPerProc;i++){
			if ( networks[i] != NULL ) {
				status = CPXNETfreeprob (envs[p], &networks[i]);
				if ( status ) {
					fprintf (stderr, "CPXNETfreeprob failed, error code %d.\n", status);
				}
			}
		}
	}
	/* Free up the CPLEX environment, if necessary */
	for(int p=0; p<myargs.processors;p++){
		if ( envs[p] != NULL ) {
			status = CPXcloseCPLEX (&envs[p]);

			/* Note that CPXcloseCPLEX produces no output,
				so the only way to see the cause of the error is to use
				CPXgeterrorstring.  For other CPLEX routines, the errors will
				be seen if the CPXPARAM_ScreenOutput indicator is set to CPX_ON. */

			if ( status ) {
			char  errmsg[CPXMESSAGEBUFSIZE];
				fprintf (stderr, "Could not close CPLEX environment.\n");
				CPXgeterrorstring (envs[p], status, errmsg);
				fprintf (stderr, "%s", errmsg);
			}
		}
	}
}
