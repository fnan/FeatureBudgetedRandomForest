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

Code for paper "Feature-Budgeted Random Forest" ICML 2015. If you have any questions please contact Feng Nan (fnan@bu.edu).
Please cite us:
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
   
INSTALL: 
The code was built on Visual Studio 2012. Boost library is needed.
To install Boost, please refer to: http://sourceforge.net/projects/boost/files/boost/1.59.0/

INPUTS:
The required input arguments are:
-f: Number of features in the data sets 

Optional arguments are:
-m: number of classes(labels).
-t: number of trees for random forest.
-p: number of processors/threads to use.
-i: impurity function: threashold-Pairs(0,default), entropy(1), powers(2).
-a: alpha, threashold for impurity function (default: 0).
-o: set to 0 if oob samples are NOT used (default); set to 1 if oob samples are used ; set to 2 if oob samples AND validation samples are used for determining node label for test points.
-c: cost vector input file name, one feature cost per row. Assume uniform feature cost if cost input file not supplied.
-R: run ranking (default classification).

data_input_file format: 
Classification - each line represent an example, label comes in the first column, followed by featureID:featureValue with white spaces in between.
Ranking - each line represent an query-document example, rank label comes in the first column, followed by qid:(query ID), followed by featureID:featureValue with white spaces in between.

cost_input_file format:
Cost of each feature appears in each row. Number of rows equal number of features.

OUTPUTS:
Expected error for classification or precision for ranking + expected feature cost.
errPrec1: predict test example label for each tree using training data distribution at the corresponding leaf, take majority label as final prediction.
errPrec2: predict test example label for each tree using out-of-bag data distribution at the corresponding leaf, take majority label as final prediction.
errPrec3: cumulate training data distributions at the corresponding leaves for all trees, take label with highest probability

EXAMPLE:
BudgetRF.exe -f 50 -t 10 -p 5 -m 2 -o 0 mbne_tiny_tr mbne_tiny_tv mbne_out_tv mbne_tiny_te mbne_out_te
BudgetRF.exe -f 519 -t 10 -p 2 -m 2 -o 0 -c yahoo_cs_cost yahoo_cssm_tr yahoo_cssm_te yahoo_outTe

!IMPORTANT! always provide the training input data file name, followed by validation input data file name, validation output file name, test input file name, test output file name.
The validation input/output files are optional.
