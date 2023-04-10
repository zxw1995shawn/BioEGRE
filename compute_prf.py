#export DATA="CHEMPROT"
#export MAX_LENGTH=128
#export BATCHSIZE=8
#export NUM_EPOCHS=3
#export NUM_CHOSEN_NEIGHBORS=32
#export NUM_GPNN_OUTPUT_NODE=8
#export CLASSIFIER_DROPOUT=0.0
#export USE_CLS=False
#export NUM_GPNN_LAYERS=1-0
#export MODEL+BioELECTRA-GPNN-alldata
#export TEXT_RESULTS_DIR=${DATA}-${MAX_LENGTH}-${BATCHSIZE}-${NUM_EPOCHS}-${NUM_CHOSEN_NEIGHBORS}-${NUM_GPNN_OUTPUT_MODE}-${CLASSIFIER_OUTPUT}-${}
import numpy as np
import sklearn
import sklearn.metrics as sm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score


pred=[]
# print("Input the model file name, for example: ./output/chemprot-128-16-50-0.0-BioELECTRA-GPNN-alldata/")
#./output/GAD-128-8-3-32-8-0.0-False-1-0-BioELECTRA-GPNN-GAD/
#./output/chemprot-128-8-3-32-8-0.0-False-1-0-BioELECTRA-GPNN-alldata/
# model_file_name = input()
model_file_name = "/tmp/zhengxw/output/euadr-1-128-16-20-BioELECTRA/"

with open(model_file_name + "test_results.txt", 'r') as f:
    for line in f.readlines():
        line_list = line.strip('\n').split('\t') #对于一行line，根据制表符，分割成若干个组件，存在line_list中，line_list[0]是index，line_list[1]是prediction
        if line_list[0]=="index" and line_list[1]=="prediction":#说明是第一行
            continue
        else:#说明不是text_results的第一行
            pred.append(line_list[1])#把text_results里面索引对应的预测的分类结果对应地append到pred里面
print("pred.size:",len(pred))
real=[]
# with open("../datasets/RE/GAD/1/test.tsv", 'r') as g:
with open("../datasets/RE/euadr/1/test.tsv", 'r') as g:
    for line in g.readlines():
        line_list = line.strip('\n').split('\t')
        if line_list[0]=="index" and line_list[1]=="sentence" and line_list[2]=="label":
            continue
        real.append(line_list[2])#把真实值存到real列表中。
print("real.size:",len(real))

confusion_matrix=sm.confusion_matrix(real,pred)

print("confusion_matrix:")
print(confusion_matrix)

if 'chemprot' not in model_file_name:
    p1=float(confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1]))
    r1=float(confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0]))
    f1=(2*p1*r1)/(p1+r1)
    print("precision of label 1:",p1)
    print(" recall   of label 1:",r1)
    print(" f-score  of label 1:",f1)

if 'chemprot' in model_file_name:
    prf=classification_report(real,pred,target_names=['Flase','CPR:3','CPR:4','CPR:5','CPR:6','CPR:9'])
    TP=confusion_matrix[1][1]+confusion_matrix[2][2]+confusion_matrix[3][3]+confusion_matrix[4][4]+confusion_matrix[5][5]

    FP1=confusion_matrix[1][1]+confusion_matrix[2][1]+confusion_matrix[3][1]+confusion_matrix[4][1]+confusion_matrix[5][1]+confusion_matrix[0][1]
    FP2=confusion_matrix[1][2]+confusion_matrix[2][2]+confusion_matrix[3][2]+confusion_matrix[4][2]+confusion_matrix[5][2]+confusion_matrix[0][2]
    FP3=confusion_matrix[1][3]+confusion_matrix[2][3]+confusion_matrix[3][3]+confusion_matrix[4][3]+confusion_matrix[5][3]+confusion_matrix[0][3]
    FP4=confusion_matrix[1][4]+confusion_matrix[2][4]+confusion_matrix[3][4]+confusion_matrix[4][4]+confusion_matrix[5][4]+confusion_matrix[0][4]
    FP5=confusion_matrix[1][5]+confusion_matrix[2][5]+confusion_matrix[3][5]+confusion_matrix[4][5]+confusion_matrix[5][5]+confusion_matrix[0][5]

    FN1=confusion_matrix[1][1]+confusion_matrix[1][2]+confusion_matrix[1][3]+confusion_matrix[1][4]+confusion_matrix[1][5]+confusion_matrix[1][0]
    FN2=confusion_matrix[2][1]+confusion_matrix[2][2]+confusion_matrix[2][3]+confusion_matrix[2][4]+confusion_matrix[2][5]+confusion_matrix[2][0]
    FN3=confusion_matrix[3][1]+confusion_matrix[3][2]+confusion_matrix[3][3]+confusion_matrix[3][4]+confusion_matrix[3][5]+confusion_matrix[3][0]
    FN4=confusion_matrix[4][1]+confusion_matrix[4][2]+confusion_matrix[4][3]+confusion_matrix[4][4]+confusion_matrix[4][5]+confusion_matrix[4][0]
    FN5=confusion_matrix[5][1]+confusion_matrix[5][2]+confusion_matrix[5][3]+confusion_matrix[5][4]+confusion_matrix[5][5]+confusion_matrix[5][0]

    microp=TP/(FP1+FP2+FP3+FP4+FP5)
    print("microp:",microp)

    micror=TP/(FN1+FN2+FN3+FN4+FN5)
    print("micror:",micror)

    microf1=(2*microp*micror)/(microp+micror)
    print("microf1:",microf1)
else:
    prf=classification_report(real,pred,target_names=['0','1'])

print(prf)
