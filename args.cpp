#include "args.h"
#include "getopt.h"

int get_args(int argc, char* argv[], args_t& args) {
  int index, c, i=0;
  int maxSplit=0;

  // option arguments
  opterr = 0;
  while ((c = getopt(argc, argv, "a:b:d:i:t:m:p:f:k:c:g:u:o:v:I:W:S:R")) != -1)
	  switch (c) {
      case 'a': args.alpha = atof(optarg); break;
      case 'm': args.num_c = atoi(optarg); break;
      case 'd':	args.depth = atoi(optarg); break;
	  case 'i': args.loss = (alg_loss)atoi(optarg); break;
      case 't':	args.trees = atoi(optarg); break;
      case 'p':	args.processors = atoi(optarg); break;
      case 'o': args.oob = atoi(optarg); break;
      case 'f':	args.features = atoi(optarg)+1; 
		  args.sensors=args.features;
		  for(int j=0;j<args.features;j++){
			  args.Costs.push_back(1.0);
			  args.CostSensors.push_back(1.0);
			  args.Costgroup.push_back(j);
		  }
		  break;
      case 'k':	args.kfeatures = atoi(optarg); break;
      case 'c': args.cost_file = (optarg); break; //cost file is optional
	  case 'g': args.costgroup_file = (optarg); break; //costgroup file is optional
      case 'R': args.alg = ALG_RANK; break;
	  case 'u': args.prune = atof(optarg); break;
      case 'v': args.verbose = 1; break;
	  case 'b': args.learning_rate_base=atof(optarg); break;
	  case 'I': args.incre = atoi(optarg); break;
	  case 'W': args.maxNumWeakLearners=atoi(optarg); break;
	  case 'S': args.bagging_rate = atof(optarg); break;
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
