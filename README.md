# 利用機器學習找尋比特幣的交易訊號
### Team members: 吳振豪

## 1. Motivation
[Patel et al (2015) Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques, *Expert Systems with Applications*, 42 (1) 259-268](https://reader.elsevier.com/reader/sd/pii/S0957417414004473?token=D40E832435591209217C8601EAA18FA53C19B293BB1B90576B33C375B057CD96D3F94C7A83BF24535EBC29EDFA697EE9&originRegion=us-east-1&originCreation=20221006112146)

![](https://i.imgur.com/Rck3Wki.png)

加密貨幣是我第一個接觸的投資項目，我認為以技術分析指標來抓住一些交易的訊號及反轉的點位能建立起更好的交易策略，因此想藉由機器學習的一些模型更精確的辨識出**指標的變化情形**和**總體價格的漲跌趨勢**之間的統整。

## 2. Data 
![](https://i.imgur.com/2hAkYdK.png)
串接Binance API獲取BTC/USDT在2019/01/01~2022/11/30之OHLC資料(interval為4HOUR)，並分別算出對應的技術指標

### label
1. 價格(label)依一段時間後的漲跌標上對應的label: 0為跌，1為漲。  
比前一筆高=長，比前一筆低=跌

### feature
historical return(拿t=n時前14個time frame的return%建)

指標則依各自的性質編出對應的漲跌關係(feature)



| indicator                 | +1(漲)                        | -1(跌)                        |
|:------------------------- | ----------------------------- |:----------------------------- |
| SMA                       | price > SMA                   | price < SMA                   |
| WMA                       | price > WMA                   | price < WMA                   |
| EMA                       | price > EMA                   | price < EMA                   |
| KD                        | KD<20                         | KD>80                         |
| KD(值介於20~80)           | K>D                           | K<D                           |
| William R%                | William R%<-80                | William R%>-20                |
| William R%(值介於-20~-80) | t<sub>WilliamR%</sub> > t-1<sub>WilliamR%</sub> | t<sub>WilliamR%</sub> > t-1<sub>WilliamR%</sub> |
| MACD                      | t<sub>DIF-MACD</sub> > t-1<sub>DIF-MACD</sub>   | t<sub>DIF-MACD</sub> > t-1<sub>DIF-MACD</sub>   |
| DMI                       | D+>D-                         | D+<D-                         |
| RSI                       | RSI<20                        | RSI>80                        |
| RSI(值介於20~80)          | t<sub>RSI</sub> > t-1<sub>RSI</sub>             | t<sub>RSI</sub> < t-1<sub>RSI</sub>             |
| CCI                       | CCI<-200                      | CCI>200                       |
| CCI(值介於-200~200)       | t<sub>CCI</sub> > t-1<sub>CCI</sub>             | t<sub>CCI</sub> < t-1<sub>CCI</sub>             |
| Momentum                  | +                             | -                             |

![](https://i.imgur.com/TljBMLq.png)




### PCA
![](https://i.imgur.com/n0mT0FZ.png)



大致來說都混在一坨了，4層的PCA只有大概80%的解釋能力，可見各指標並沒有一個共同的趨勢或是特性在。

再來想透過ML綜合各式指標的預測效果，以及是否有篩選的機制

## 3. Formulation


從技術指標 ${\bf X}=\{x_1,x_2,\cdots,x_p\}$ 預測下一時間(4小時後)股價的漲跌($y$)
train: 2019-01-07 00:00 ~ 2021-09-30 20:00
test: 2021-10-01 00:00 ~ 2022-11-30 20:00
![](https://i.imgur.com/qOM9vdh.png)


### 1. Benchmark: 單一指標預測股價的準確度(confusion matrix)
accuracy(=(tp+tn)/(tn+tp+fn+fp))  
recall=tp/(tp+fn)  
precision=tp/(tp+fp)  
f1-score=2 * precision * recall / (precision + recall)  
![](https://i.imgur.com/bnEjFYC.png)
![](https://i.imgur.com/gy59KIi.png)

基本上各式的技術指標的效果皆不太好(多數在5成左右)，仰賴單一指標預測隔天漲跌跟盲猜差不多

### 2.return(tuning=RandomizedSearchCV)
![](https://i.imgur.com/MiOyKss.png=350x)
![](https://i.imgur.com/QIe9xW1.png=350x)


### 3. 各式Classifier(tuning=RandomizedSearchCV)

* with 10 indicator  
![](https://i.imgur.com/I3qbgjJ.png=340x)
![](https://i.imgur.com/3fx8WVS.png=340x)

和單一技術指標或一般的return也都差不多

## 4. Analysis
回測採用幣安的手續費設定(0.1%)  
交易策略為一次只買進/賣出一顆比特幣(依交易訊號而定)，且倉位最多就是一顆比特幣  
若之前已持有，訊號給買進: 不動作  
若之前已持有，訊號給賣出: 平倉持有的倉位，改空  
若之前已賣出，訊號給賣出: 不動作  
若之前已賣出，訊號給買進: 平倉持有的倉位，改多  

(SMA cross用兩條SMA(快線和慢線)，快線和慢線黃金交叉則買進，死亡交叉則賣出)
### 策略比較
(buy&hold:-60.572641%)
| strategy                         | score/acc | Return (%) | Sharpe Ratio | Max. Drawdown (%) | Win Rate (%) | # Trades |
|:-------------------------------- |:--------- |:---------- |:------------ |:----------------- |:------------ |:-------- |
| Baseline: SMAcross               | -         | -6.15      |              | -27.69            | 32.35        | 204      |
| Baseline:single indicator(DMI)   | 0.508     | 15.76      | 0.53         | -13.13            | 27.98        | 193      |
| return(no tuning:AdaBoost)       | 0.510     | 36.30      | 1.02         | -16.19            | 50.21        | 703      |
| return(no tuning:KNN)            | 0.505     | 14.72      | 0.49         | -16.07            | 50.00        | 1184     |
| 10 indicator(no tuning:AdaBoost) | 0.495     | 311.20     | 5.23         | -2.67             | 75.32        | 697      |
| 10 indicator(tuning:KNN)         | 0.4898    | 181.62     | 3.70         | -4.03             | 55.53        | 1122     |


![](https://i.imgur.com/qZyHFlB.png)

## 5. Conclusion
1. 預測下一時間的漲跌在較簡單的模型上可能沒有辦法預測得很好，但回測的結果也顯示只要大方向對了即可(亦即不一定要盯著每根k棒去猜下一次的漲跌，抓到關鍵的點位或趨勢即可)
2. 綜合各式指標後確實能建立更好的交易訊號(相較於單一指標或是return而言，也可贏過簡單的策略)
3. 未來方向可以考慮最佳化/減少交易訊號(目前還是太過頻繁交易導致利潤被手續費吃掉)，建立更穩定的模型(參數最佳化等)，更加彈性的交易策略等
## Reference

## Supplimentary materials

**指標公式**

close price $P_{t}$，$t$ on a 4 hour basis
$\text{High}_t$ =time t的最高價， $\text{Low}_t$ =time t的最低價
$n=14$(time window)，下列指標皆使用14天的衡量

$\textbf{SMA}(14)$(價格的算術平均) 

$$
\frac{\sum\limits_{k=t-14}^t P_k}{14}
$$

$\textbf{WMA}(14)$ (價格對經過天數的平均(越遠越低))

$$
\frac{\sum\limits_{k=t-14}^t P_k*(t-k)}{\frac{14+(14+1)}{2}}
$$ 

$\textbf{EMA}(14)$ (納入之前的EMA(但影響會以指數下滑))

$$
P_t\times\frac{2}{14+1}+\text{EMA}_{t-1}\times(1-\frac{2}{14+1})
$$ 

$\textbf{RSI}$ (相對的強弱比例(上漲/下跌))  

$$
100-100*\frac{1}{1+\frac{\text{MA(14)}\_{U\_t}}{\text{MA(14)}\_{D\_t}}}
$$

$(U_t = \text{max}(\text{close}\_t-\text{close}\_{t-1},0))$  
$(D_t = \text{max}(\text{close}\_{i-t}-\text{close}_{t},0))$  

$\textbf{KD}$ (股價相對的高低位置)

$$
K_t = K_{t-1}* (2/3) + \text{RSV}\_t*(1/3)\\
D_t = D_{t-1}* (2/3) + K_t*(1/3)
$$

$\text{RSV}\_t = (\frac{P_t — L_{14}}{H_{14} — L_{14}})*100$
(現在股價處於14日內最高價和最低價間的位置)  
$H_{14}$ = highest price in past 14 days  
$L_{14}$ = Lowest price in past 14 days  

$\textbf{William %R}$(類似K值，但以最高價-現在價格計算)

$$
\frac{H_{14}-P_t}{H_{14}-L_{14}}\times-100
$$

$\textbf{MACD}$(14日DIF的EMA)

$$
\text{DIF}=\text{EMA}(14)-\text{EMA}(28)(長短\text{EMA}再平滑)
$$

$\textbf{DMI}$(上升/下降趨勢的強弱)

$$
D+=(\frac{\text{Smoothed +DM}}{\text{Smoothed TR}}\times100)\\
D-=(\frac{\text{Smoothed -DM}}{\text{Smoothed TR}}\times100)
$$

$\text{TR}=\text{max}(\text{High}\_t-\text{Close}\_{t-1},\text{High}\_t-\text{Low}\_{t-1},\text{Low}\_t- \text{Close}\_{t-1})$  
$\text{+DM}=\text{max}(\text{High}\_t-\text{High}\_{t-1},0)$  
$\text{-DM}=\text{max}(\text{Low}\_{t-1}-\text{Low}\_{t},0)$  
$\text{Smoothed}+/-\text{DM}=\sum\limits_{t=1}^{14}\text{DM}-\frac{\sum\limits_{t=1}^{14}\text{DM}}{14}+\text{DM}_t$  
(smoothed的TR也是用上述公式)

$\textbf{CCI}$(衡量市場是否超出一定的分布範圍)

$$
\frac{\text{Typical Price-MA}}{0.015\times\text{Mean Deviation}}
$$

$\text{Typical Price}=\sum\limits_{i=1}^{14}((\text{High}+\text{Low}+\text{Close})\div 3)$
$\text{MA} = (\sum\limits_{i=1}^{14}\text{Typical Price})\div 14$
$\text{Mean Deviation}=(\sum\limits_{i=1}^{14}\text{|Typical Price-MA|})\div 14$

$\textbf{momentum}$ 

$$
P_t-P_{t-14}
$$  
  
**Backtest的詳細結果**  
(SMAcross)  
![](https://i.imgur.com/QpMQExY.png)
![](https://i.imgur.com/6llk4Pa.png)
![](https://i.imgur.com/nIM2rFZ.png)
![](https://i.imgur.com/P068gvb.png)
![](https://i.imgur.com/ot89gc7.png)
![](https://i.imgur.com/o1zEuW4.png)

