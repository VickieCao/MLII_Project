import numpy as np
import pandas as pd
import talib
from sklearn import decomposition
import matplotlib.pyplot as plt
import itertools


def HMA(real, timeperiod=16):
    return talib.WMA(2 * talib.WMA(real, int(timeperiod / 2.0))
           - talib.WMA(real, timeperiod), int(np.sqrt(timeperiod)))


def Chaikin_oscillator(High, Low, Close, Volume, n=3, m=9):
    ADL = ((Close - Low) - (High - Close)) / (High - Low) * Volume + 1
    return talib.EMA(ADL, n) - talib.EMA(ADL, m)


def Chaikin_vol(High, Low, timeperiod=10):
    Range = High - Low
    return ((pd.DataFrame(talib.EMA(Range, timeperiod)).pct_change() - 1) * 100.0).values


def normalize(ts, start=None, end=None):
    maximum = np.max(ts.ix[start:end])
    minimum = np.min(ts.ix[start:end])
    return (ts - minimum) / (maximum - minimum)


def feature_analysis(data=None, feature=None, pca_components=None, graph=False,
                     start=None, end=None):
    X = data[feature].values.reshape(-1, len(feature))
    X_train = data[feature].ix[start:end].values.reshape(-1, len(feature))
    pca = decomposition.KernelPCA(n_components=pca_components)
    pca.fit(X_train)
    pcaresult = pca.transform(X)
    # print(pca.components_)
    ica = decomposition.FastICA(n_components=pca_components)
    ica.fit(X_train)
    icaresult = ica.transform(X)
    pcaresult = (pcaresult.T.reshape(pca_components, -1))
    icaresult = (icaresult.T.reshape(pca_components, -1))
    for n in range(pca_components):
        data['%s-pcomponent' % str(n+1)] = pcaresult[n]
        data['%s-icomponent' % str(n+1)] = icaresult[n]
    # print(pca.explained_variance_ratio_.cumsum())
    if graph is True:
        for j in range(1, pca_components+1):
            plt.clf()
            data['%i-pcomponent' % j].plot()
            plt.legend()
            plt.plot()
            plt.show()

    return data

def calculate_features(data: pd.DataFrame, normalization=False, train_data:list=None,
                       start=None, end=None):
    Open = data['Open'].values
    High = data['High'].values
    Low = data['Low'].values
    Close = data['Close'].values
    Volume = data['Volume'].values
    data['ret'] = data['Close'].pct_change() * 100.0
    data['ret_2'] = data['Close'].pct_change().shift() * 100.0
    data['ret_3'] = data['Close'].pct_change().shift(2) * 100.0
    data['ret_4'] = data['Close'].pct_change().shift(3) * 100.0
    data['ret_5'] = data['Close'].pct_change().shift(4) * 100.0
    data['ret_ratio'] = (data['ret'] / data['ret_5'] - 1) * 100.0
    data['log_ret'] = (np.log(data['Close'])).diff() * 100.0
    data['gap'] = ((data['Open'] - data['Close'].shift()) / data['Open'] * 100.0)
    data['gap2'] = ((data['Open'] - data['Close'].shift()) / data['Open'] * 100.0).shift()
    data['gap3'] = ((data['Open'] - data['Close'].shift()) / data['Open'] * 100.0).shift(2)
    data['gap4'] = ((data['Open'] - data['Close'].shift()) / data['Open'] * 100.0).shift(3)
    data['gap5'] = ((data['Open'] - data['Close'].shift()) / data['Open'] * 100.0).shift(4)
    data['hl'] = ((data['High'] - data['Low']) / data['Open'] * 100.0)
    data['hl2'] = ((data['High'] - data['Low']) / data['Open'] * 100.0).shift()
    data['hl3'] = ((data['High'] - data['Low']) / data['Open'] * 100.0).shift(2)
    data['hl4'] = ((data['High'] - data['Low']) / data['Open'] * 100.0).shift(3)
    data['hl5'] = ((data['High'] - data['Low']) / data['Open'] * 100.0).shift(4)
    data['oc'] = ((data['Close'] - data['Open']) / data['Open'] * 100.0)
    data['oc2'] = ((data['Close'] - data['Open']) / data['Open'] * 100.0).shift()
    data['oc3'] = ((data['Close'] - data['Open']) / data['Open'] * 100.0).shift(2)
    data['oc4'] = ((data['Close'] - data['Open']) / data['Open'] * 100.0).shift(3)
    data['oc5'] = ((data['Close'] - data['Open']) / data['Open'] * 100.0).shift(4)
    data['MA_short'] = talib.EMA(data['Close'].values, 10)
    data['MA_long'] = talib.EMA(data['Close'].values, 120)
    data['MA_ratio'] = (data['MA_short'] / data['MA_long'] - 1) * 100.0
    data['MA2_short'] = talib.EMA(data['Close'].values, 10)
    data['MA2_long'] = talib.EMA(data['Close'].values, 60)
    data['MA2_ratio'] = (data['MA2_short'] / data['MA2_long'] - 1) * 100.0
    data['vol_long'] = pd.rolling_std(data['Close'], 30)
    data['vol_short'] = pd.rolling_std(data['Close'], 15)
    data['vol_ratio'] = (data['vol_short'] / data['vol_long'] - 1) * 100.0
    data['EMA'] = (Close / talib.EMA(Close, 5) -1)  * 100.0
    data['EMA_long'] = (Close / talib.EMA(Close, 60) -1)  * 100.0
    data['RSI'] = talib.RSI(data['Close'].values) / 100.0
    data['MOM'] = talib.MOM(data['Close'].values, timeperiod=14) / 100.0
    data['MACD_vfast'], data['MACD_signal_vfast'], data['MACD_hist'] = \
        talib.MACD(data['Close'].values, fastperiod=4, slowperiod=9, signalperiod=3)
    data['MACD_fast'], data['MACD_signal_fast'], _ = \
        talib.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD_slow'], _, _ = talib.MACD(data['Close'].values, fastperiod=25, slowperiod=50)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'].values,
                                                                      fastperiod=30, slowperiod=65, signalperiod=22)
    data['ATR'] = talib.ATR(High, Low, Close, timeperiod=28)
    data['ADX_vlong'] = talib.ADX(High, Low, Close, timeperiod=120)
    data['ADX_long'] = talib.ADX(High, Low, Close, timeperiod=28)
    data['ADX_short'] = talib.ADX(High, Low, Close, timeperiod=14)
    data['TSF_short'] = talib.TSF(data['Close'].values, timeperiod=14)
    data['TSF_long'] = talib.TSF(data['Close'].values, timeperiod=28)
    data['TSF_ratio'] = (data['TSF_short'] / data['TSF_long'] - 1) * 100.0
    data['BBand_up'], data['BBand_mid'], data['BBand_low'] = talib.BBANDS(data['Close'].values,
                                                                          timeperiod=20)
    data['BBand_width'] = (data['BBand_up'] / data['BBand_low'] - 1) * 100.0
    data['HMA_short'] = HMA(data['Close'].values, timeperiod=9)
    data['HMA_long'] = HMA(data['Close'].values, timeperiod=60)
    data['HMA_ratio'] = (data['HMA_short'] / data['HMA_long'] - 1) * 100.0
    data['HMA_ret'] = HMA(data['Close'].values, 100)
    # data['HMA_ret'] = data['HMA_ret'].pct_change()
    data['OBV'] = talib.OBV(Close, Volume)
    data['mean'] = pd.rolling_mean(data['ret'], 10)
    data['std'] = pd.rolling_std(data['ret'], 10)
    data['skewness'] = pd.rolling_skew(data['ret'], 10)
    data['kurtosis'] = (pd.rolling_kurt(data['ret'], 10) - 3)
    data['STOCHk'], data['STOCHd'] = talib.STOCH(High, Low, Close, fastk_period=28, slowk_period =3, slowd_period =3)
    data['STOCHRSId'], data['STOCHRSIk'] = talib.STOCHRSI(Close)
    data['Chaikin_vol'] = Chaikin_vol(High, Low)
    data['Chaikin_oscillator'] = Chaikin_oscillator(High, Low, Close, Volume)
    data['PDI'] = talib.PLUS_DI(High, Low, Close, timeperiod=14)
    data['MDI'] = talib.MINUS_DI(High, Low, Close, timeperiod=14)
    data['DI'] = data['ADX_short'] - data['PDI'] + data['MDI']
    # train_data  = ['ret', 'ret_2', 'ret_3', 'ret_4', 'ret_5', 'vol_ratio', 'hl', 'oc', 'gap']
    # 'ret_2', 'ret_3', 'ret_4', 'ret_5']
    # data = include_VIX(data)
    data.replace(np.nan, 0, inplace=True)
    if normalization is True:
        for feature in data.columns:

            if feature not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Product',
                               'log_ret', 'ret', 'ret_2', 'ret_3', 'ret_4', 'ret_5', 'Date']:
                data[feature] = (normalize(data[feature], start=start, end=end))

    if train_data is None:
        # train_data = ['MACD_vfast', 'vol_ratio', 'oc', 'hl', 'ret', 'ADX_short', 'MA_ratio', 'MA2_ratio',
        #               'RSI', 'skewness', 'kurtosis', 'mean', 'std']
        train_data = ['oc', 'vol_ratio', 'hl', 'ret']
        # train_data = ['MACD_vfast', 'vol_ratio', 'oc', 'hl', 'gap', 'ret', 'ADX_short', 'BBand_width', 'MA_ratio',
    #                   'RSI', 'skewness', 'kurtosis', 'mean', 'std'] # most original
        # train_data = ['MACD_vfast', 'vol_ratio', 'oc', 'hl', 'gap', 'ret',
        #               'ADX_short', 'BBand_width', 'MA_ratio', 'RSI', 'skewness', 'kurtosis', 'mean', 'std']
    data = feature_analysis(data, feature=train_data, pca_components=len(train_data), start=start, end=end)

    return data