#include <algorithm>
#include "impurity.h"
#include <boost\numeric\ublas\vector.hpp>
#include <boost\bind.hpp>
#include <boost\thread\thread.hpp>

using namespace std;

extern vector<vector<int> > fInds;

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
double impurityHP(int num_c, boost::numeric::ublas::vector<int>& c_tmp, double alpha){
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
double impurityEntropy(int num_c, boost::numeric::ublas::vector<int>& c_tmp, double alpha){
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
	return -imp;
}

bool mysortf(const tupleW* tk1, const tupleW* tk2, int f) {
  return tk1->features[f] < tk2->features[f];
}
//min-max splits based on impurity measure
bool impurity_splitW_noMiss(data_t data,  int& f_split, double& v_split,  double imp, boost::numeric::ublas::vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double)) {
	int NF=myargs.features;
	int num_c=myargs.num_c;
	double alpha=myargs.alpha;
	f_split = -1;
	double min = MY_DBL_MAX, cf;
	int n = data.size(), i,j ;
	double imp_l=0.0, imp_r=0.0, imp_m=0.0, imp_max=0.0; //impurity on the left

	double mind;
	double v_split_d;
	for (int f = 1; f < NF; f++) { //for each feature
		cf=myargs.Costs[f];
		sort(data.begin(), data.end(), boost::bind(mysortf, _1,_2, f));
		boost::numeric::ublas::vector<int> c_l(num_c,0); //number of examples in each class on the left
		boost::numeric::ublas::vector<int> c_r(c_total); //number of examples in each class on the right
		
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
bool impurity_splitE_noMiss(data_t data,  int& f_split, double& v_split,  double imp, boost::numeric::ublas::vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double)) {
	int NF=myargs.features;
	int num_c=myargs.num_c;
	double alpha=myargs.alpha;
	f_split = -1;
	double min = MY_DBL_MAX, cf;
	int n = data.size(), i,j ;
	double imp_l=0.0, imp_r=0.0, imp_m=0.0, imp_e=0.0; //impurity on the left

	double mind;
	double v_split_d;
	for (int f = 1; f < NF; f++) { //for each feature
		cf=myargs.Costs[f];
		sort(data.begin(), data.end(), boost::bind(mysortf, _1,_2, f));
		boost::numeric::ublas::vector<int> c_l(num_c,0); //number of examples in each class on the left
		boost::numeric::ublas::vector<int> c_r(c_total); //number of examples in each class on the right
		
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

