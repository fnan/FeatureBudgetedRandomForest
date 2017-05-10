# FeatureBudgetedRandomForest
The MIT License (MIT)

Copyright (c) 2015 Feng Nan (fnan@bu.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code for papers "Pruning Random Forests for Prediction on a Budget" NIPS 2016 and "Feature-Budgeted Random Forest" ICML 2015.
If you have any questions please contact Feng Nan (fnan@bu.edu).
Please cite us:
@inproceedings{DBLP:conf/nips/NanWS16,
  author    = {Feng Nan and
               Joseph Wang and
               Venkatesh Saligrama},
  title     = {Pruning Random Forests for Prediction on a Budget},
  booktitle = {Advances in Neural Information Processing Systems 29: Annual Conference
               on Neural Information Processing Systems 2016, December 5-10, 2016,
               Barcelona, Spain},
  pages     = {2334--2342},
  year      = {2016},
  crossref  = {DBLP:conf/nips/2016},
  url       = {http://papers.nips.cc/paper/6250-pruning-random-forests-for-prediction-on-a-budget},
  timestamp = {Fri, 16 Dec 2016 19:45:58 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/nips/NanWS16},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
@inproceedings{icml2015_nan15,
   Publisher = {JMLR Workshop and Conference Proceedings},
   title="Feature-Budgeted Random Forest",
   Author={Nan, Feng and Wang, Joseph and Saligrama, Venkatesh},
   year="2015",
   Booktitle="Proceedings of the 32nd International Conference on Machine Learning (ICML-15)",
   Editor={David Blei and Francis Bach},
   pages="1983-1991",
   url="http://jmlr.org/proceedings/papers/v37/nan15.pdf"
} 

Some parts of the code taken from the RT-Rank project: https://sites.google.com/site/rtranking/home
implemented by Ananth Mohan (mohana@go.wustl.edu) and Zheng Chen, under the supervision of Dr. Kilian Weinberger.
   
# INSTALL: 
The code was built on Visual Studio 2012/2013. 
Required libraries:
1. Cplex: https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/ (used to solve network flow problems, 32-bit version may cause unpredicted behaviour so please use 64-bit version.)
2. lpsolve: https://sourceforge.net/projects/lpsolve/ (used to define LP problems)


# INPUTS:
## The required input arguments are:
* -f: Number of features in the data sets 

## Optional arguments are:
* -m: number of classes(labels).
* -t: number of trees for random forest. (must be multiples of number of threads used)
* -p: number of processors/threads to use.
* -i: impurity function: threashold-Pairs(0,default), entropy(1), powers(2).
* -a: alpha, threashold for impurity function (default: 0).
* -o: set to 0 if oob samples are NOT used (default); set to 1 if oob samples are used ; set to 2 if oob samples AND validation samples are used for determining node label for test points.
* -c: cost vector input file name, one feature cost per row. Assume uniform feature cost if cost input file not supplied.
* -R: run ranking (default classification).
* -k: number of randomly selected features to build each tree (default -1: use all features).
## Added for Pruning:
* -u: error-feature cost trade off parameter. 0.0 if no pruning; >0.0 to perform pruning. Larger value leads to heavier pruning.
* -b: learning rate base parameter for dual ascend step in pruning; larger value leads to larger step size. default: 500.
* -g: costgroup file, optional. Use this option when a group of feature corresponds to the same **sensor** and **shares** an acquisition cost. It specifies the grouping of features so that we don't double count the cost within the groups.



## data_input_file format: 
### Classification - each line represent an example, label comes in the first column, followed by featureID:featureValue with white spaces in between.
### Ranking - each line represent an query-document example, rank label comes in the first column, followed by qid:(query ID), followed by featureID:featureValue with white spaces in between.

## cost vector input file format:
Cost of each feature appears in each row. Number of rows equal number of features.

## costgroup file format:
For example, suppose there are only 2 sensors. Each sensor gives a multi-dimensional measurement. 
Suppose Sensor1 has acquisition cost of 0.1 and produces 2 features; Sensor2 has acquisition cost of 0.2 and produces 3 features. 
The data input thus has 5 features (training matrix size = n x 5, n is the number of data points).  
The cost input file for the -c option should look like:
0.1
0.1
0.2
0.2
0.2

The costgroup file should be:
1
1
2
2
2


# OUTPUTS:
1. Expected error for classification or precision for ranking + expected feature cost.
..* errPrec1: predict test example label for each tree using training data distribution at the corresponding leaf, take majority label as final prediction.
..* errPrec2: predict test example label for each tree using out-of-bag data distribution at the corresponding leaf, take majority label as final prediction.
..* errPrec3: cumulate training data distributions at the corresponding leaves for all trees, take label with highest probability

2. featMatrix files: contains the feature usage matrix: (i,j)th element is the number of times example i uses feature j.

3. proba_pred files: contains the predicted probabilities of each example in each class.

4. tree file: tree structure file

5. Standard output for pruning iterations: primal objective | dual objective | duality gap.

# EXAMPLE:
To build the forest:
```
BudgetRF.exe -f 50 -t 10 -p 5 -m 2 -o 0 mbne_tiny_tr mbne_tiny_tv mbne_out_tv mbne_tiny_te mbne_out_te
```
To prune the forest:
```
BudgetRF.exe -f 50 -t 10 -p 5 -m 2 -o 0 -u 0.02 mbne_tiny_tr mbne_tiny_tv mbne_out_tv mbne_tiny_te mbne_out_te
```
To build the forest for ranking:
```
BudgetRF.exe -f 519 -t 10 -p 2 -m 2 -o 0 -c yahoo_cs_cost yahoo_cssm_tr yahoo_cssm_te yahoo_outTe
```
To use a customized cost file:
```
BudgetRF.exe -f 50 -t 10 -p 5 -c cost_file_name -m 2 -o 0 mbne_tiny_tr mbne_tiny_tv mbne_out_tv mbne_tiny_te mbne_out_te
```
To use a customized cost file with group cost:
```
BudgetRF.exe -f 50 -t 10 -p 5 -c cost_file_name -g cost_group_file_name -m 2 -o 0 mbne_tiny_tr mbne_tiny_tv mbne_out_tv mbne_tiny_te mbne_out_te
```

# !IMPORTANT! 
Always provide the training input data file name, followed by validation input data file name, validation output file name, test input file name, test output file name.
The validation input/output files are optional. The tree file needs to be available to run pruning. So first run BudgetRF without pruning (to build the forest and generate tree file), then with pruning. 

