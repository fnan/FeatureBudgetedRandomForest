//parse command line arguments
#ifndef AM_RT_ARGS_H
#define AM_RT_ARGS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include "getopt.h"
#include "tupleW.h"
#include <vector>
using namespace std;

typedef vector<tupleW*> data_t;
typedef vector< vector<tupleW*> > vec_data_t;
typedef vector<double> preds_t;
typedef vector< vector<double> > vec_preds_t;

enum alg_t { ALG_CLASSIFICATION,ALG_RANK};
enum alg_loss { ALG_HP, ALG_ENTROPY, ALG_POWER };
//enum alg_pred { ALG_MEAN, ALG_MODE };

struct args_t
{
  int features;
  int trees;
  int processors;
  double alpha;
  int depth;
  int kfeatures;
  char* train_file;
  char* cost_file;
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
  vector<double> Costs;
  int oob; //include out of bag samples in tree leaves
};

static void init_args(args_t& a) {
  a.processors = 1;
  a.trees = 1;
  a.alpha = 0.0;
  a.features = 0;
  a.depth = 1000;
  a.num_test = 1;
  a.kfeatures = -1;
  a.alg = ALG_CLASSIFICATION;
  a.verbose = 1;
  a.rounds = 1;
  a.loss = ALG_HP;
  a.missing = 0;
  a.num_c=2;
  a.cost_file=NULL;
  a.oob=0; // set to 0 if oob samples are NOT used (default); set to 1 if oob samples are used ; set to 2 if oob samples AND validation samples are used for determining node label for test points
}


static int get_args(int argc, char* argv[], args_t& args) {
  int index, c, i=0;

  // option arguments
  opterr = 0;
  while ((c = getopt (argc, argv, "a:d:i:t:m:p:f:k:r:c:o:vR")) != -1)
    switch (c) {
      case 'a': args.alpha = atof(optarg); break;
      case 'm': args.num_c = atoi(optarg); break;
      case 'd':	args.depth = atoi(optarg); break;
	  case 'i': args.loss = (alg_loss)atoi(optarg); break;
      case 't':	args.trees = atoi(optarg); break;
      case 'p':	args.processors = atoi(optarg); break;
      case 'o': args.oob = atoi(optarg); break;
      case 'f':	args.features = atoi(optarg)+1; 
		  for(int j=0;j<args.features;j++)
			  args.Costs.push_back(1.0);
		  break;
      case 'k':	args.kfeatures = atoi(optarg); break;
      case 'r': args.rounds = atoi(optarg); break;
      case 'c': args.cost_file = (optarg); break; //cost file is optional
      case 'R': args.alg = ALG_RANK; break;
      case 'v': args.verbose = 1; break;
      case '?':
	if (optopt == 'c')
	  fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	else if (isprint (optopt))
	  fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	else
	  fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
	return 0;
      default:
	return 0;
      }

  // non option arguments
  if (argc-optind < 3)
    return 0;
  for (index = optind; index < argc; index++) {
    if (i==0) args.train_file = argv[index];
    else if (i%2) args.test_files.push_back(argv[index]);
    else args.test_outs.push_back(argv[index]);
    i++;
  }
  args.num_test = args.test_files.size();

  return 1;
}

#endif
