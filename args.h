//parse command line arguments
#ifndef AM_RT_ARGS_H
#define AM_RT_ARGS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "tupleW.h"

using std::vector;

typedef vector<tupleW*> data_t;
typedef vector< vector<tupleW*> > vec_data_t;
typedef vector<double> preds_t;
typedef vector< vector<double> > vec_preds_t;

enum alg_t { ALG_CLASSIFICATION,ALG_RANK,ALG_BOOST_MAXSPLIT, ALG_BOOST_EXPSPLIT};
enum alg_loss { ALG_HP, ALG_ENTROPY, ALG_POWER };
//enum alg_pred { ALG_MEAN, ALG_MODE };

struct args_t
{
  int features;
  int sensors;
  int trees;
  int processors;
  double alpha;
  int depth;
  int gateDepth;
  int kfeatures;
  char* train_file;
  char* cost_file;
  char* costgroup_file;
  char* tree_file;
  vector<char*> test_files;
  vector<char*> test_outs;
  int num_test;
  alg_t alg;
  int verbose;
  int rounds;
  alg_loss loss;
  int missing;
  int ntra;
  int num_c; //number of classes
  double prune; //if negative, train tree; if positive, serve as lambda trade-off parameter if pruneMethod is 0 or 1; serve as number of internal nodes to keep if pruneMethod is 2
  vector<double> Costs;
  vector<int> Costgroup;
  vector<double> CostSensors;
  int oob; //include out of bag samples in tree leaves
  int pruneMethod; //0: (default) prune as a forest, 1: prune individual tree, 2: cost-complexity pruning
  bool analysis; //0: (default) not perform feature analysis, 1: perform feature analysis, requires tree input files
  int incre; //incremental (mini-batch) pruning size. Default 0.
  int maxNumWeakLearners; //maximum number of weak learners in gating function
  double bagging_rate; //subsampling ratio to build each tree
  double learning_rate_base; // learning rate in the primal-dual pruning algorithm = 1/(t+learning_rate_base)
};

static void init_args(args_t& a) {
  a.processors = 1;
  a.trees = 1;
  a.alpha = 0.0;
  a.features = 0;
  a.sensors =0;
  a.depth = 1000;
  a.gateDepth = 3;
  a.num_test = 1;
  a.kfeatures = -1;
  a.alg = ALG_CLASSIFICATION;
  a.verbose = 0;
  a.rounds = 1;
  a.loss = ALG_HP;
  a.missing = 0;
  a.num_c=2;
  a.prune=-1;
  a.cost_file=NULL;
  a.costgroup_file=NULL;
  a.oob=0; // set to 1 if oob samples are used for calculating pruning error; set to 2 if oob samples are used for calculating pruning AND determining node label for test points
  a.pruneMethod=0;
  a.analysis=0;
  a.incre=0;
  a.maxNumWeakLearners=0; 
  a.bagging_rate = 1.0;
  a.learning_rate_base = 500;
};

int get_args(int argc, char* argv[], args_t& args);

#endif
