import os
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str, help='test_results.txt: Model generated predictions.')
parser.add_argument('--answer_path', type=str, help='test.tsv file: Tab-seperated. One example per a line. True labels at the 3rd column.')
parser.add_argument('--task', type=str, default="binary", help='default:binary, possible other options:{chemprot}')
args = parser.parse_args()


testdf = pd.read_csv(args.answer_path, sep="\t", index_col=0)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)


#binary
if args.task == "binary":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [int(v[1]) for v in pred[1:]]
    print("pred:",pred)
    print("pred_class:",pred_class)
    print("testdf_label",testdf["label"])
    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])
    results = dict()
    results["f1 score"] = f[1]
    results["recall"] = r[1]
    results["precision"] = p[1]
    results["specificity"] = r[0]

# chemprot
#/home/zhengxw/chemprot-128-64-50-32-4-0.0-False-0-0-BioELECTRA-GPNN-alldata
# micro-average of 5 target classes
# see "Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction (Mehryary, 2018)" for details
if args.task == "chemprot":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    print("pred:",pred)
    pred_class = [np.argmax(v) for v in pred]
    print("pred_class:",pred_class)
    print("lenpred:",len(pred))
    str_to_int_mapper = dict()

    for i,v in enumerate(sorted(testdf["label"].unique())):
        str_to_int_mapper[v] = i
    test_answer = [str_to_int_mapper[v] for v in testdf["label"]]
    print("test_answer:",test_answer)
    print("lentestans:",len(test_answer))
    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, labels=[0,1,2,3,4,5], average="micro")
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p

for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))

#

