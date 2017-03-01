#include <algorithm>
#include <functional>

#include "impurity.h"

using namespace std;

extern vector<vector<int> > fInds;

vector<int> sort_indexes(const vector<double> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}

void sort_data_by_feature(vector<int>& location,  vector<int> dataCount, vector<int> invertIdx, int f){
	int cur=0;
    for (int i = 0; i < fInds[f].size(); i ++){
        int z = fInds[f][i];
		int loc=invertIdx[z]; //loc is -1 if data z is not present in current data
        for (int j = 0; j < dataCount[z]; j ++)
			location[cur++]=loc;
	}
}

//threahold-Pairs impurity
//inputs - num_c: number of classes
//		 - c_tmp: weights of each class
//		 - alpha: threashold
double impurityHP(int num_c, vector<int>& c_tmp, double alpha){
	vector<double> c_th(c_tmp.size(), 0.0);
	int i,j;
	double imp=0.0,imp_ij=0.0;
	for (i=0;i<num_c;i++)
		c_th[i]= (c_tmp[i]< alpha)? 0 : c_tmp[i]-alpha;
	for (i=0;i<num_c-1;i++){
		for (j=i+1;j<num_c;j++){
			imp_ij=c_th[i]*c_th[j]-alpha*alpha;
			if (imp_ij>0)
				imp+=imp_ij;
		}
	}
	return imp;
}

//entropy impurity
//inputs - num_c: number of classes
//		 - c_tmp: weights of each class
//		 - alpha: threashold
double impurityEntropy(int num_c, vector<int>& c_tmp, double alpha){
	int i,sumTmp=0;
	double imp=0.0,prob_i=0.0;
	for(i=0;i<num_c;i++)
		sumTmp+=c_tmp[i];
	
	if(sumTmp==0) return 0;

	for(i=0;i<num_c;i++){
		if(c_tmp[i]!=0){
			prob_i=(double)c_tmp[i]/sumTmp;
			imp+=prob_i*log(prob_i);
		}
	}
	return (-imp < alpha)? 0 : -imp;
}

//compute the deviance (variance) of the set
double impurityDeviance(vector<double>& targets){
	double sum = std::accumulate(targets.begin(), targets.end(),0.0);
	double mean = sum / targets.size();
	vector<double> diff(targets.size());
	std::transform(targets.begin(),targets.end(),diff.begin(), [mean](double x) {return x - mean;});
	return std::inner_product(diff.begin(),diff.end(),diff.begin(),0.0);
}

//compute the mean squared of the set
double impurityMeanSq(vector<double>& targets){
	double sum = std::accumulate(targets.begin(), targets.end(),0.0);
	double mean = sum / targets.size();
	return mean*mean;
}

bool mysortf(const tupleW* tk1, const tupleW* tk2, int f) {
  return tk1->features[f] < tk2->features[f];
}
//min-max splits based on impurity measure
bool impurity_splitW_noMiss(data_t data,  int& f_split, double& v_split,  double imp, vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, vector<int>&,double),vector<int> featureSet) {
	int NF=myargs.features;
	int num_c=myargs.num_c;
	double alpha=myargs.alpha;
	f_split = -1;
	double min = MY_DBL_MAX, cf;
	int n = data.size(), i,j,f;
	double imp_l=0.0, imp_r=0.0, imp_m=0.0, imp_max=0.0; //impurity on the left

	double mind;
	double v_split_d;
	for (int fi = 0; fi < featureSet.size(); fi++) { //for each feature
		f=featureSet[fi];
		cf=myargs.Costs[f];
		sort(data.begin(), data.end(), std::bind(mysortf, std::placeholders::_1, std::placeholders::_2, f));
		vector<int> c_l(num_c,0); //number of examples in each class on the left
		vector<int> c_r(c_total); //number of examples in each class on the right
		
		mind = MY_DBL_MAX;
		//assume no missing data
		for( i=0;i<n-1;i++){
			c_l[data[i]->label]++;
			c_r[data[i]->label]--;
			// do not consider splitting here if data is the same as next
			if (data[i]->features[f] == data[i+1]->features[f])
				continue;
			imp_l=(*impurityHandle)(num_c, c_l, alpha);
			imp_r=(*impurityHandle)(num_c, c_r, alpha);
			imp_max=(imp_l < imp_r) ? imp_r: imp_l;
			if(imp_max<mind){
				mind=imp_max;
				v_split_d = (data[i]->features[f] + data[i+1]->features[f])/2;
			}
		}

		if ((imp-mind>0.0000001) && (cf/(imp-mind) < min)) {
			min = cf/(imp-mind);
			f_split = f;
			v_split = v_split_d;
		}
	}	
	return min != MY_DBL_MAX;
}

//expected splits based on impurity measure
bool impurity_splitE_noMiss(data_t data,  int& f_split, double& v_split,  double imp, vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, vector<int>&,double),vector<int> featureSet) {
	int NF=myargs.features;
	int num_c=myargs.num_c;
	double alpha=myargs.alpha;
	f_split = -1;
	double min = MY_DBL_MAX, cf;
	int n = data.size(), i,j,f ;
	double imp_l=0.0, imp_r=0.0, imp_m=0.0, imp_e=0.0; //impurity on the left

	double mind;
	double v_split_d;
	for (int fi = 0; fi < featureSet.size(); fi++) { //for each feature
		f=featureSet[fi];
		cf=myargs.Costs[f];
		 sort(data.begin(), data.end(), std::bind(mysortf, std::placeholders::_1, std::placeholders::_2, f));
		vector<int> c_l(num_c,0); //number of examples in each class on the left
		vector<int> c_r(c_total); //number of examples in each class on the right
		
		mind = MY_DBL_MAX;
		//assume no missing data
		for( i=0;i<n-1;i++){
			c_l[data[i]->label]++;
			c_r[data[i]->label]--;
			// do not consider splitting here if data is the same as next
			if (data[i]->features[f] == data[i+1]->features[f])
				continue;
			imp_l=(*impurityHandle)(num_c, c_l, alpha);
			imp_r=(*impurityHandle)(num_c, c_r, alpha);
			imp_e=((i+1)*imp_l+(n-i-1)*imp_r)/n;
			if(imp_e<mind){
				mind=imp_e;
				v_split_d = (data[i]->features[f] + data[i+1]->features[f])/2;
			}
		}

		if ((imp-mind>0.0000001) && (cf/(imp-mind) < min)) {
			min = cf/(imp-mind);
			f_split = f;
			v_split = v_split_d;
		}
	}	
	return min != MY_DBL_MAX;
}


//void evalFeatureSplitsPerProc(const vector<vector<double>>& dataMatrix, const vector<double>& dataTargets,  vector<double>& impReduction, vector<double>& v_splits,  double imp, int startInd, int endInd){
void evalFeatureSplitsPerProc(const vector<vector<double>>& dataMatrix, const vector<double>& dataTargets, vector<double>& impReduction, vector<double>& v_splits,  searchSplitParam& param){
	int startInd=param.start; int endInd=param.end;

	double var=0.0, varL=0.0, varR=0.0, sumTarget,sumTargetSq, sumTargetR,sumTargetRSq, sumTargetL=0.0, sumTargetLSq=0.0,varReduceTmp=0.0; //impurity on the left
	double weightL, weightR;
	int i, n = dataTargets.size();
	vector<int> sortedInd;
	vector<double> dataTargetsSq(n,0.0);

	sumTarget=accumulate(dataTargets.begin(), dataTargets.end(),0.0);
	std::transform(dataTargets.begin(),dataTargets.end(), dataTargets.begin(), dataTargetsSq.begin(), std::multiplies<double>());
	sumTargetSq = accumulate(dataTargetsSq.begin(), dataTargetsSq.end(),0.0);
	var = sumTargetSq/n - (sumTarget*sumTarget/(n*n));

	for (int fi = startInd; fi < endInd; fi++) { //for each feature
		sumTargetR=sumTarget;
		sumTargetL=0.0;
		sumTargetRSq = sumTargetSq;
		sumTargetLSq = 0.0;
		sortedInd=sort_indexes(dataMatrix[fi]);
		//assume no missing data
		for( i=0;i<n-1;i++){
			sumTargetL+=dataTargets[sortedInd[i]];
			sumTargetR-=dataTargets[sortedInd[i]];
			sumTargetLSq+=dataTargetsSq[sortedInd[i]];
			sumTargetRSq-=dataTargetsSq[sortedInd[i]];
			// do not consider splitting here if data is the same as next
			if (dataMatrix[fi][sortedInd[i]] == dataMatrix[fi][sortedInd[i+1]])
				continue;
			varL=sumTargetLSq/(i+1)-sumTargetL*sumTargetL/((i+1)*(i+1));
			varR=sumTargetRSq/(n-i-1)-sumTargetR*sumTargetR/((n-i-1)*(n-i-1));
			if(param.alg==ALG_BOOST_MAXSPLIT)
				varReduceTmp=((i+1)*(var-varL) < (n-i-1)*(var-varR)) ? (n-i-1)*(var-varR): (i+1)*(var-varL);
			else
				varReduceTmp=(i+1)*(var-varL) + (n-i-1)*(var-varR);

			if(impReduction[fi] < varReduceTmp){
				impReduction[fi] = varReduceTmp;
				v_splits[fi] = (dataMatrix[fi][sortedInd[i]] + dataMatrix[fi][sortedInd[i+1]])/2;
			}
		}
	}
}

void logLoss(data_t& data, const vector<double>& currentPred, double& loss){
	loss=0.0;
	for(int i=0; i< data.size();i++){
		if(data[i]->label==0){
			loss+=log(1+exp(currentPred[i]));
			data[i]->pred=-1/(1+exp(-currentPred[i])); //negative gradient
		}
		else{
			loss+=log(1+exp(-currentPred[i]));
			data[i]->pred=1/(1+exp(currentPred[i]));
		}
	}
}

void pseudoLogLoss(data_t& data, const vector<double>& currentPred, double& loss){
	loss=0.0;
	for(int i=0; i< data.size();i++){
		if(data[i]->psLabel==0){
			loss+=log(1+exp(currentPred[i]))*data[i]->weight;
			data[i]->pred=-data[i]->weight/(1+exp(-currentPred[i])); //negative gradient
		}
		else{
			loss+=log(1+exp(-currentPred[i]))*data[i]->weight;
			data[i]->pred=data[i]->weight/(1+exp(currentPred[i]));
		}
	}
}
