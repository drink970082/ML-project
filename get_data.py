from talib import abstract  # talib 算技術指標
from jupyterthemes import jtplot
import copy
# import mplfinance as mpf  # 畫K線跟技術指標
import talib
from binance.client import Client
import pandas as pd
import numpy as np

"""
get public and private keys, use Binance API to get historical Kline
"""


def get_klines(interval,start,end):
    with open('key.txt') as f: # key.txt 存API的公私鑰
        keys = f.readlines()
    keys = [key.rstrip('\n') for key in keys]
    client = Client(keys[0], keys[1])
    data = client.get_historical_klines("BTCUSDT",
                                        interval,
                                        start,
                                        end)
    df = pd.DataFrame(data, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume',
                                    'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol',
                                    'takerBuyQuoteVol', 'ignore'])
    df['dateTime'] = df['dateTime']/1000  # dateTime多3個0
    df['dateTime'] = pd.to_datetime(df['dateTime'], unit='s')
    return df


"""
extract OHLC from Kline
"""


def get_OHLC(df):
    OHLC = df[["open", "high", "low", "close", "volume"]]
    OHLC.index = df['dateTime']
    OHLC.index.name = 'Date'
    OHLC.columns = ["Open", "High", "Low", "Close", "Volume"]
    OHLC["Open"] = OHLC["Open"].astype(float)
    OHLC["High"] = OHLC["High"].astype(float)
    OHLC["Low"] = OHLC["Low"].astype(float)
    OHLC["Close"] = OHLC["Close"].astype(float)
    OHLC["Volume"] = OHLC["Volume"].astype(float)
    # OHLC.to_csv('OHLC.csv',index = 1)
    # OHLC.head()
    return OHLC


"""
use talib to generate tech index from OHLC
"""


def get_tech(OHLC,period):
    df_talib = copy.deepcopy(OHLC)
    df_talib.columns = ["open", "high", "low", "close", "volume"]
    wma = abstract.WMA(df_talib, timeperiod=period)
    ema = abstract.EMA(df_talib, timeperiod=period)
    sma = abstract.SMA(df_talib, timeperiod=period)
    rsi = abstract.RSI(df_talib, timeperiod=period)  # RSI
    kd = abstract.STOCH(df_talib, fastk_period=period)  # KD
    macd = abstract.MACD(df_talib, fastperiod=period, slowperiod=period*2, signalperiod=9)
    cci = abstract.CCI(df_talib, timeperiod=period)
    mdi = abstract.MINUS_DI(df_talib, timeperiod=period)
    pdi = abstract.PLUS_DI(df_talib, timeperiod=period)
    mtm = abstract.MOM(df_talib, timeperiod=period)
    willr = abstract.WILLR(df_talib, timeperiod=period)
    tech_idx = pd.DataFrame.from_dict(ema)
    tech_idx.columns = ['EMA']
    tech_idx['WMA'] = wma.values
    tech_idx['SMA'] = sma.values
    tech_idx['RSI'] = rsi.values
    tech_idx['STOCH_K'] = kd['slowk'].values
    tech_idx['STOCH_D'] = kd['slowd'].values
    tech_idx['DIF'] = macd['macd'].values
    tech_idx['MACD'] = macd['macdsignal'].values
    tech_idx['DIF-MACD'] = macd['macdhist'].values
    tech_idx['CCI'] = cci.values
    tech_idx['MDI'] = mdi.values
    tech_idx['PDI'] = pdi.values
    tech_idx['MTM'] = mtm.values
    tech_idx['WILLR'] = willr.values
    tech_idx['close'] = OHLC["Close"]
    return tech_idx


"""
generate label
"""


def get_label(OHLC):
    OHLC_label = copy.deepcopy(OHLC)
    window = 6
    OHLC_label['return'] = OHLC_label.Close.pct_change()
    # OHLC_label['vol'] = OHLC_label.Close.pct_change().rolling(
    #     window).std()*(window**0.5)
    # OHLC_label['std'] = OHLC_label.Close.pct_change().rolling(window).std()
    OHLC_label['label_2'] = 0  # 只有漲跌
    # OHLC_label['label_3'] = 0  # 加上盤整
    # OHLC_label['thres'] = 0
    # OHLC_label['thres'] = OHLC_label['Close']*(1.5*OHLC_label['std'])
    OHLC_label['label_2'][OHLC_label['Close'].gt(
        OHLC_label['Close'].shift(-1))] = 1  # 比前一個大
    OHLC_label['label_2'][OHLC_label['Close'].lt(
        OHLC_label['Close'].shift(-1))] = 0  # 比前一個小
    # OHLC_label['label_2'][OHLC_label['label_2']==0] = 1  # 比前一個大
    # OHLC_label['label_3'][OHLC_label['Close']
    #                       > (OHLC_label['Close'].rolling(window).mean()
    #                          + OHLC_label['thres'].rolling(1).sum())] = 1
    # OHLC_label['label_3'][OHLC_label['Close']
    #                       < (OHLC_label['Close'].rolling(window).mean()
    #                          - OHLC_label['thres'].rolling(1).sum())] = -1
    print("label:")
    print(OHLC_label['label_2'].value_counts())
    # print(OHLC_label['label_3'].value_counts())
    # OHLC_label.iloc[20:].head()
    return OHLC_label


"""
generate features
"""


def get_features(tech_idx):
    feature_idx = copy.deepcopy(tech_idx)
    for i in ['EMA', 'WMA', "SMA"]:
        feature_idx[i][(tech_idx[i] < tech_idx['close'])] = 1
        feature_idx[i][(tech_idx[i] > tech_idx['close'])] = 0
    feature_idx['RSI'][(tech_idx["RSI"] > tech_idx["RSI"].shift(-1))] = 1
    feature_idx['RSI'][(tech_idx["RSI"] <= tech_idx["RSI"].shift(-1))] = 0
    feature_idx['RSI'][(tech_idx["RSI"] > 80)] = 0
    feature_idx['RSI'][(tech_idx["RSI"] < 20)] = 1
    feature_idx['STOCH_K'][(tech_idx['STOCH_K'] > tech_idx['STOCH_D'])] = 1
    feature_idx['STOCH_K'][(tech_idx['STOCH_K'] < tech_idx['STOCH_D'])] = 0
    feature_idx['STOCH_K'][(tech_idx['STOCH_D'] > 80)] = 0
    feature_idx['STOCH_K'][(tech_idx['STOCH_D'] < 20)] = 1
    feature_idx['KD'] = feature_idx['STOCH_K']
    feature_idx.drop(['STOCH_D', 'STOCH_K'], axis=1, inplace=True)
    feature_idx['MACD'][(tech_idx["DIF-MACD"] >
                        tech_idx["DIF-MACD"].shift(-1))] = 1
    feature_idx['MACD'][(tech_idx["DIF-MACD"] <
                        tech_idx["DIF-MACD"].shift(-1))] = 0
    feature_idx.drop(['DIF-MACD', 'DIF'], axis=1, inplace=True)
    feature_idx['CCI'][(tech_idx["CCI"] > tech_idx["CCI"].shift(-1))] = 1
    feature_idx['CCI'][(tech_idx["CCI"] < tech_idx["CCI"].shift(-1))] = 0
    feature_idx['CCI'][(tech_idx["CCI"] > 200)] = 0
    feature_idx['CCI'][(tech_idx["CCI"] < -200)] = 1
    feature_idx['MDI'][(tech_idx['PDI'] > tech_idx['MDI'])] = 1
    feature_idx['MDI'][(tech_idx['PDI'] < tech_idx['MDI'])] = 0
    feature_idx['DMI'] = feature_idx['MDI']
    feature_idx.drop(['MDI', 'PDI'], axis=1, inplace=True)
    feature_idx['MTM'][(tech_idx['MTM'] > 0)] = 1
    feature_idx['MTM'][(tech_idx['MTM'] < 0)] = 0
    feature_idx['WILLR'][(tech_idx['WILLR'] > tech_idx['WILLR'].shift(-1))] = 1
    feature_idx['WILLR'][(tech_idx['WILLR'] < tech_idx['WILLR'].shift(-1))] = 0
    feature_idx['WILLR'][(tech_idx['WILLR'] > -20)] = 0
    feature_idx['WILLR'][(tech_idx['WILLR'] < -80)] = 1
    feature_idx.drop(['close'], axis=1, inplace=True)
    feature_idx.dropna(how='any', inplace=True)
    feature_idx = feature_idx[:-1]
    # feature_cnt = feature_idx.apply(pd.Series.value_counts)
    # print(feature_cnt)
    #feature_idx.tail(10)
    return feature_idx