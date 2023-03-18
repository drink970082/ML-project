import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def confusion(OHLC_hasFeature,feature_idx_sel,splitpoint,display=False):
    tp_train = np.zeros(10)
    fp_train = np.zeros(10)
    fn_train = np.zeros(10)
    tn_train = np.zeros(10)
    tp_test = np.zeros(10)
    fp_test = np.zeros(10)
    tn_test = np.zeros(10)
    fn_test = np.zeros(10)
    acc_train = np.zeros(10)
    recall_train = np.zeros(10)
    precision_train = np.zeros(10)
    acc_test = np.zeros(10)
    recall_test = np.zeros(10)
    precision_test = np.zeros(10)
    f1_test = np.zeros(10)
    for idx,i in enumerate(feature_idx_sel.columns):
        cm = confusion_matrix(OHLC_hasFeature[:splitpoint], feature_idx_sel[i][:splitpoint])
        tn, fp, fn, tp = cm.ravel()
        tp_train[idx] = tp/(tn+tp+fn+fp)
        fp_train[idx] = fp/(tn+tp+fn+fp)
        fn_train[idx] = fn/(tn+tp+fn+fp)
        tn_train[idx] = tn/(tn+tp+fn+fp)
        acc_train[idx]=(tp+tn)/(tn+tp+fn+fp)
        recall_train[idx]=tp/(tp+fn)
        precision_train[idx]=tp/(tp+fp)
        cm = confusion_matrix(OHLC_hasFeature[splitpoint:], feature_idx_sel[i][splitpoint:])
        tn, fp, fn, tp = cm.ravel()
        tp_test[idx] = tp/(tn+tp+fn+fp)
        fp_test[idx] = fp/(tn+tp+fn+fp)
        fn_test[idx] = fn/(tn+tp+fn+fp)
        tn_test[idx] = tn/(tn+tp+fn+fp)
        acc_test[idx]=(tp+tn)/(tn+tp+fn+fp)
        recall_test[idx]=tp/(tp+fn)
        precision_test[idx]=tp/(tp+fp)
        if display:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            print(i)
            disp.plot()  
            plt.show()
    f1_train=2 * precision_train * recall_train / (precision_train + recall_train)
    f1_test=2 * precision_test * recall_test / (precision_test + recall_test)
    matrix_basic={
        "tp(train)": tp_train,
        "tn(train)": tn_train,
        "fp(train)":fp_train,
        "fn(train)":fn_train,
        "tp(test)":tp_test,
        "tn(test)":tn_test,
        "fp(test)":fp_test,
        "fn(test)":fn_test
    }
    confusion_basic = pd.DataFrame(matrix_basic,index=["EMA","WMA","SMA","RSI","MACD","CCI","MTM","WILLR","KD","DMI"])
    # confusion_basic
    matrix_cal={
        "accuracy(train)": acc_train,
        "accuracy(test)": acc_test,
        "Recall(train)":recall_train,
        "Recall(test)":recall_test,
        "Precision(train)":precision_train,
        "Precision(test)":precision_test,
        "F1-score(train)":f1_train,
        "F1-score(test)":f1_test
    }
    confusion_score = pd.DataFrame(matrix_cal)
    confusion_score.index = ["EMA","WMA","SMA","RSI","MACD","CCI","MTM","WILLR","KD","DMI"]
    # confusion_score
    return confusion_basic,confusion_score

def training(X_train,X_test,y_train,y_test,method,method_name):
    train_score= []
    test_score =[]
    with tqdm(total=len(method), desc="progress") as pbar:
        for i in range(len(method)):
            clf = method[i].fit(X_train,y_train)
            # print(method_name[i])
            #print('training:%f'%clf.score(X_train, y_train))
            #print('testing:%f'%clf.score(X_test, y_test))
            train_score.append(clf.score(X_train, y_train))
            test_score.append(clf.score(X_test, y_test))
            #print('-----------------------------------------')
            pbar.set_postfix({'method': method_name[i]})
            pbar.update(1)
    table = {
        "train score":train_score,
        "test score":test_score
    }
    # print("without tuning_return")
    df_method = pd.DataFrame(table,index = method_name)
    return df_method