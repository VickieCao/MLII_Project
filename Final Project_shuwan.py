#Load stock data

import pandas_datareader as web
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import pdist

from sklearn.linear_model import LinearRegression

Tickers_bank=['JPM','BAC','C','WFC','GS','MS','USB','TD','PNC']
Tickers_consumer=['PEP','KO','MCD','YUM','SBUX','CMG']
Tickers=Tickers_bank+Tickers_consumer
start_date='2012-04-01'
end_date='2014-04-01' #must before back-tracking time
stock_data= web.get_data_yahoo(Tickers, start=start_date, end=end_date)

#Compute technical indicators

def calculateFeatures(Tickers,stock_data):
    #need to test each of these
    data = {}
    data['ret'] = stock_data['Adj Close'].pct_change() * 100.0
    data['ret_5'] = stock_data['Adj Close'].pct_change().shift(4) * 100.0
    data['ret_ratio'] = (data['ret'] / data['ret_5'] - 1) * 100.0
    data['gap'] = ((stock_data['Open'] - stock_data['Adj Close'].shift()) / stock_data['Open'] * 100.0)
    data['hl'] = ((stock_data['High'] - stock_data['Low']) / stock_data['Open'] * 100.0)
    data['MA_short'] = stock_data['Adj Close'].apply(lambda x: talib.EMA(x, 10))
    data['MA_long'] = stock_data['Adj Close'].apply(lambda x: talib.EMA(x, 60))
    data['MA_ratio'] = (data['MA_short'] / data['MA_long'] - 1) * 100.0
    data['vol_short'] = stock_data['Adj Close'].rolling(10).std()
    data['vol_long'] = stock_data['Adj Close'].rolling(60).std()
    data['vol_ratio'] = (data['vol_short'] / data['vol_long'] - 1) * 100.0
    data['MOM'] = stock_data['Adj Close'].apply(lambda x: talib.MOM(x, timeperiod=10) / 100)
    data['skewness'] = data['ret'].rolling(10).skew()
    data['kurtosis'] = data['ret'].rolling(10).kurt() - 3
    data['RSI'] = stock_data['Adj Close'].apply(lambda x: talib.RSI(x) / 100)
    data=pd.concat(data.values(),axis=1,keys=data.keys())
    stock_data=pd.concat([stock_data,data],axis=1)
    stock_data.dropna(inplace=True)

    #reconstruct data, indexed by labels
    stock_data.columns = stock_data.columns.swaplevel(1, 0)
    stock_data.sort_index(axis=1,inplace=True)
    return stock_data

def reformData(Tickers,data):
    df=pd.DataFrame()
    for ticker in Tickers:
        df=pd.concat([df,data[ticker]],axis=0)
    return df
           
    
#Use PCA to find principal components
def getPCA(Tickers,stock_data,n_comp=3,normalize=True,variance=True):

    stock_data=calculate_features(Tickers,stock_data)
    stock_data=stock_data.replace([np.inf,-np.inf],np.nan)
    stock_data.fillna(method='ffill',inplace=True)
    if normalize:
        scaler=MinMaxScaler()
        data_scaled=scaler.fit_transform(np.array(stock_data))
        data_scaled=pd.DataFrame(data_scaled,columns=stock_data.columns)
        data_scaled=data_scaled.stack().unstack(level=0).T
    else:
        data_scaled=data_scaled.stack().unstack(level=0).T
    pca=PCA(n_components=n_comp).fit(np.array(data_scaled))
    print(np.sum(pca.explained_variance_ratio_))
    stock_PCA=pd.DataFrame(pca.transform(np.array(data_scaled)))
    stock_PCA.index=stock_data.index
    stock_PCA=stock_PCA.T.stack(level=0).T
    return stock_PCA
    #First three components can explain around 70% variance, and we think this is great.

#Stock clustering based on principal components
def clusterStock(Tickers,stock_data,h):
    stock_PCA=getPCA(Tickers,stock_data,3)
    #use hierarchy clustering

    #use correlation as the similarity measurement
    stock_PCA.sort_index(axis=1,inplace=True)

    Distance_mat1=pdist(np.array(stock_PCA[0].T),'correlation')
    Distance_mat2=pdist(np.array(stock_PCA[1].T),'correlation')
    Distance_mat3=pdist(np.array(stock_PCA[2].T),'correlation')
    D=(Distance_mat1+Distance_mat2+Distance_mat3)/3 #take an average of them as the total measure
    D=(D+D.T)/2
    D=squareform(D)
    tree=hierarchy.linkage(D,'complete')
    plt.figure()
    dn=hierarchy.dendrogram(tree)
    plt.title('Dendrogram, complete linkage ')
    plt.ylabel('Height')
    plt.show()
    
    print(stock_PCA.columns.get_level_values(1))
    #Select Pairs
    #['GS','MS'],['KO','PEP']
    pairs=[['GS','MS'],['KO','PEP']]

def getSpread(pairs,window=60,end_date='2014-06-01'):
#     '''pairs should be in format [[p1,p2],[p1,p2]]
#     return spread, index:date,value:spread'''

    tmp=datetime.datetime.strptime(end_date, '%Y-%m-%d')
    delta = datetime.timedelta(days=window)
    start_date=(tmp-delta).strftime('%Y-%m-%d')
    spread={}
    #download data, compute return
    for i,pair in enumerate(pairs):
        stock_data=web.get_data_yahoo(pair, start=start_date, end=end_date)['Adj Close']
        ret = stock_data.pct_change() * 100.0
        ret=ret[1:]
        days=ret.shape[0]
        X=ret[pair[0]].values.reshape((days,1))
        y=ret[pair[1]].values.reshape((days,1))
        lr=LinearRegression().fit(X,y)
        spread['pair'+str(i+1)]=(y-lr.predict(X)).reshape(days)
    spread=pd.DataFrame(spread,index=stock_data.index[1:])
    return spread
    
    
    
    
    
    
    
    
    
    
    
    



