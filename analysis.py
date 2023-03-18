from backtesting import Backtest, Strategy #引入回測和交易策略功能
from backtesting.test import SMA #從test子模組引入繪製均線功能
from backtesting.lib import crossover #從lib子模組引入判斷均線交會功能
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class SmaCross(Strategy):
    def init(self):
        self.fast_line = self.I(SMA, self.data.Close, 5)
        self.slow_line = self.I(SMA, self.data.Close, 15)

    def next(self):
        if not self.position:
            if crossover(self.fast_line, self.slow_line):
                self.buy(size=1)
            elif crossover(self.slow_line, self.fast_line):
                self.sell(size=1)
        
        elif crossover(self.fast_line, self.slow_line) and self.position.is_short:
            self.position.close()
            self.buy(size=1)
        elif crossover(self.slow_line, self.fast_line) and self.position.is_long:
            self.position.close()
            self.sell(size=1)

def batch_test_idc(OHLC_backtesting,feature_test,y_test,name,plot=False):
    indic=["EMA","WMA","SMA","RSI","MACD","CCI","MTM","WILLR","KD","DMI"]
    accuracy_list=[]
    return_list = [] 
    sharpe_list = []
    maxdraw_list = []
    win_list = []
    tradenum_list = []
    i=0
    for t in indic:
        i+=1
        tech_label = feature_test[t].to_numpy()
        cm = confusion_matrix(y_test, tech_label)
        tn, fp, fn, tp = cm.ravel()
        accuracy_list.append((tp+tn)/(tn+tp+fn+fp))
        class tech(Strategy):
            label = tech_label
            i=0
            def init(self):
                self.i=0
            def next(self):
                #print(self.data)
                if self.i>=len(self.label): 
                    return
                if not self.position:
                    if self.label[self.i] == 1:
                        self.buy(size=1)
                    else:
                        self.sell(size=1)
                else:
                    if self.label[self.i] == 1 and self.position.is_short:
                        self.position.close()
                        self.buy(size=1)
                    elif self.label[self.i] == 0 and self.position.is_long:
                        self.position.close()
                        self.sell(size=1)
                self.i+=1
        test_tech = Backtest(OHLC_backtesting, tech, cash=100000, commission=0.001,trade_on_close=False)
        result_tech = test_tech.run(label=tech_label)
        if i ==4 and plot:
            test_tech.plot(filename=name)
        return_list.append(result_tech["Return [%]"])
        sharpe_list.append(result_tech["Sharpe Ratio"])
        maxdraw_list.append(result_tech["Max. Drawdown [%]"])
        win_list.append(result_tech["Win Rate [%]"])
        tradenum_list.append(result_tech["# Trades"])
    table = {
        "tech indicator accuracy":accuracy_list,
        "Return [%]":return_list,
        "Sharpe Ratio": sharpe_list,
        "Max. Drawdown [%]": maxdraw_list,
        "Win Rate [%]": win_list,
        "# Trades":tradenum_list
    }
    #print("backtesting_single indicator")
    df_method = pd.DataFrame(table,index = indic)
    return df_method

def batch_test_learn(OHLC_backtesting,X_train,X_test,y_train,y_test,
                method,method_name,name,plot=False):
    score = []
    return_list = [] 
    sharpe_list = []
    maxdraw_list = []
    win_list = []
    tradenum_list = []
    for m in method:
        clf = m.fit(X_train,y_train)
        score.append(clf.score(X_test,y_test))
        Return_label = clf.predict(X_test)
        class Return(Strategy):
            label = Return_label
            i=0
            def init(self):
                self.i=0
            def next(self):
                print(self.data)
                if self.i>=len(self.label): 
                    return
                if not self.position:
                    if self.label[self.i] == 1:
                        self.buy(size=1)
                    else:
                        self.sell(size=1)
                else:
                    if self.label[self.i] == 1 and self.position.is_short:
                        self.position.close()
                        self.buy(size=1)
                    elif self.label[self.i] == 0 and self.position.is_long:
                        self.position.close()
                        self.sell(size=1)
                self.i+=1
        test_Return = Backtest(OHLC_backtesting, Return, cash=100000, commission=0.001,trade_on_close=False)
        result_Return = test_Return.run(label=Return_label)
        if plot: test_Return.plot(filename=name)
        return_list.append(result_Return["Return [%]"])
        sharpe_list.append(result_Return["Sharpe Ratio"])
        maxdraw_list.append(result_Return["Max. Drawdown [%]"])
        win_list.append(result_Return["Win Rate [%]"])
        tradenum_list.append(result_Return["# Trades"])
    table = {
        "model test score":score,
        "Return [%]":return_list,
        "Sharpe Ratio": sharpe_list,
        "Max. Drawdown [%]": maxdraw_list,
        "Win Rate [%]": win_list,
        "# Trades":tradenum_list
    }
    #print("backtesting_return (no tuning)")
    df_method = pd.DataFrame(table,index = method_name)
    return df_method