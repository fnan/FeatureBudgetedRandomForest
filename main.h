#include "BudgetForest.h"

#include <iostream>
#include <vector>
#include <cmath>

#include "getopt.h"

using namespace std;

vector<vector<int> > fInds;

// given a set of files, read them into train and test vectors
// if cost_file is specified in args, read cost vector into args.Costs
int load_data(vector<tupleW*>& train, vector< vector<tupleW*> >& test, args_t& myargs, int r) {
	int numfeatures = myargs.features;
	int missing = myargs.missing;

	fprintf(stderr, "loading training data...");
	char train_file[100];	
	if(myargs.rounds>1)
		sprintf(train_file, "%s%d",myargs.train_file,r);
	else
		sprintf(train_file, "%s",myargs.train_file);

	if (!tupleW::read_input(train, train_file, numfeatures, 1, myargs.alg==ALG_RANK))
		return 0;
	fprintf(stderr, "done\n");
	fprintf(stderr, "loading test data...");	

	for (int i=0; i<myargs.num_test; i++) {
		char test_file[100];
		if(myargs.rounds>1)
			sprintf(test_file, "%s%d",myargs.test_files[i],r);
		else
			sprintf(test_file, "%s",myargs.test_files[i]);
		vector<tupleW*> t;
		if (!tupleW::read_input(t, test_file, numfeatures, 1, myargs.alg==ALG_RANK))
			return 0;
		test.push_back(t);
	}
	fprintf(stderr, "done\n");

	fprintf(stderr, "loading feature cost...");
	if (myargs.cost_file==NULL){
		fprintf(stderr, "cost file not provided, will use default\n");
		return 1;	
	}
	ifstream input(myargs.cost_file);
	if (input.fail())
		return 0;
	string strline;
	int i=1;
	while(getline(input,strline)){
		myargs.Costs[i]=atof(strline.c_str());
		myargs.CostSensors[i]=myargs.Costs[i];
		i++;
	}
	if (i!=myargs.features){
		fprintf(stderr, "inconsistent number of features in cost_file...");
		return 0;
	}

	fprintf(stderr, "loading feature cost group ...");
	if (myargs.costgroup_file==NULL){
		fprintf(stderr, "cost file not provided, will use default\n");
		return 1;	
	}
	ifstream input2(myargs.costgroup_file);
	if (input2.fail())
		return 0;
	i=1;
	while(getline(input2,strline)){
		myargs.Costgroup[i]=atoi(strline.c_str());
		i++;
	}
	if (i!=myargs.features){
		fprintf(stderr, "inconsistent number of features in costgroup_file...");
		return 0;
	}
	/* compute the unique sensors */
	myargs.CostSensors.clear();
	myargs.CostSensors.push_back(1.0);

	myargs.sensors=1;
	for(i=1;i<myargs.features;i++){
		if(myargs.Costgroup[i]>myargs.CostSensors.size()-1){
			myargs.CostSensors.push_back(myargs.Costs[i]);
			myargs.sensors++;
		}
	}
	fprintf(stderr, "done\n");
	return 1;
}

// free all the memory we used up
void free_memory(const args_t& myargs, vector<tupleW*>& train, vector< vector<tupleW*> >& test) {
  int i;
  tupleW::delete_data(train);
  for (i=0; i < myargs.num_test; i++)
    tupleW::delete_data(test[i]);
}



void add_idx(vector<tupleW*>& train) {
        int i, N = train.size();
        for (i = 0; i < N; i ++){
                train[i]->idx=i;
        }
}


bool mysortpred2(const pair<tupleW*, int> tk1, const pair<tupleW*, int > tk2) {
  return tk1.first->features[tk1.second] < tk2.first->features[tk2.second];
}

void avgprec2(const data_t& data, vector<double>& pred1Rank,vector<double>& pred2Rank, vector<double>& pred3Rank, int topX, double& prec1, double& prec2,double& prec3){
	int n=data.size(), i,j, ind;
	int nq=0;
	double avg1=0.0, avg2=0.0, avg3=0.0;
	i=0;
	while (i<n){
		int q=data[i]->qid;
		vector<pair<double, int>> tmp1;
		vector<pair<double, int>> tmp2;
		vector<pair<double, int>> tmp3;
		for(j=0;j<n;j++)
			if(data[j]->qid ==q){
				tmp1.push_back(pair<double, int>(pred1Rank[j], data[j]->label));
				tmp2.push_back(pair<double, int>(pred2Rank[j], data[j]->label));
				tmp3.push_back(pair<double, int>(pred3Rank[j], data[j]->label));
			}
		std::sort(tmp1.begin(), tmp1.end(), [](const pair<double, int> &left, const pair<double, int> &right){ return left.first > right.first;  });
		std::sort(tmp2.begin(), tmp2.end(), [](const pair<double, int> &left, const pair<double, int> &right){ return left.first > right.first;  });
		std::sort(tmp3.begin(), tmp3.end(), [](const pair<double, int> &left, const pair<double, int> &right){ return left.first > right.first;  });
		ind=0;
		while(ind<topX && ind<tmp1.size() &&  tmp1[ind].second==1) ind++;
		if(ind==tmp1.size())
			avg1+=1.0;
		else
			avg1+=(double)ind/topX;
		ind=0;
		while(ind<topX && ind<tmp2.size() &&  tmp2[ind].second==1) ind++;
		if(ind==tmp2.size())
			avg2+=1.0;
		else
			avg2+=(double)ind/topX;
		ind=0;
		while(ind<topX && ind<tmp3.size() &&  tmp3[ind].second==1) ind++;
		if(ind==tmp3.size())
			avg3+=1.0;
		else
			avg3+=(double)ind/topX;

		i+=tmp1.size();
		nq++;
	}
	prec1=avg1/nq;
	prec2=avg2/nq;
	prec3=avg3/nq;
}

