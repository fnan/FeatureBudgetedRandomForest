#ifndef FN_IMPURITY_ML_H
#define FN_IMPURITY_ML_H

#include "tupleW.h"
#include "args.h"
#include <boost\numeric\ublas\vector.hpp>
using namespace std;

bool impurity_splitW_noMiss(data_t data, int& f_split, double& v_split, double imp, boost::numeric::ublas::vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double));
bool impurity_splitE_noMiss(data_t data, int& f_split, double& v_split, double imp, boost::numeric::ublas::vector<int>& c_total,args_t& myargs, double (*impurityHandle)(int, boost::numeric::ublas::vector<int>&,double));
double impurityEntropy(int num_c, boost::numeric::ublas::vector<int>& c_tmp, double alpha);
double impurityHP(int num_c, boost::numeric::ublas::vector<int>& c_tmp, double alpha);

#endif