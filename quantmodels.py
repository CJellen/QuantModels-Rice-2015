from __future__ import division
__metaclass__ = type
from numpy import *
from pandas import DataFrame, Series
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from pandas.io.data import DataReader
from datetime import datetime
from scipy.optimize import fmin_slsqp
from scipy.stats.mstats import winsorize 

def convert(x,old,new, newtype='int') :
    assert newtype=='int' or newtype=='str', "newtype in changedates must be int or str"
    x = datetime.strptime(str(x),old).strftime(new)
    if newtype == 'int' : return int(x)
    else : return x
        
def stock_read(filename,stock,date) :
    X = pd.read_csv(filename+'.csv',index_col=[date,stock])
    return StockTable(X,stock=stock,date=date)
    
def return_read(filename,date) :
    X = pd.read_csv(filename+'.csv',index_col=date)
    return ReturnTable(X,date=date)

def ShrinkCovs(A,delta=1/3) :
    A = DataFrame(A)
    x = sqrt(diag(A))
    corrs = ((A/x).T / x).T
    sm = sum(tril(corrs,-1))
    k = A.shape[0]
    mn = sm * 2 / (k*(k-1))
    B = (1-delta)*corrs + delta*mn
    for i in range(k) :
        B.ix[i,i] = 1
    x = (1-delta)*x + delta * mean(array(x))
    C = ((B*x).T * x).T 
    return DataFrame(C)
    
class ReturnTable(DataFrame) :
    def __init__(self,df,date='d') :
        super(ReturnTable,self).__init__(df)
        self.date = date
        self.sort_index(inplace=True)
    def __repr__(self) :
        return DataFrame.__repr__(self)
    def __str__(self) :
        return DataFrame.__str__(self)
    def stock(self,indexname='',valuename='') :
        X = self.stack()
        X = DataFrame(X,columns=[valuename])
        X.index.names=[self.date,indexname]
        return StockTable(X,date=self.date,stock=indexname)
    def write(self,filename) :
        self.to_csv(filename + '.csv',index=True)
    def changedates(self,old,new,newtype='int') :
        R = DataFrame(self).reset_index()
        R[self.date] = R[self.date].map(lambda x: convert(x,old,new,newtype))
        _ = R.set_index(self.date,drop=True,inplace=True)
        return ReturnTable(R,date=self.date)
    def plot(self,cols=False,dateformat='%Y%m') :
        R = DataFrame(self).reset_index()
        R[self.date] = R[self.date].apply(lambda x: datetime.strptime(str(x),dateformat))
        R = R.set_index(self.date)
        if not(cols) : 
            List = list(self.columns)
            if len(List) == 1 : cols = List[0]
            else : cols = List
        return R[cols].plot()
    def head(self) :
        X = DataFrame(self)
        return X.head()
    def tail(self) :
        X = DataFrame(self)
        return X.tail()
    def dropday(self) :
        self[self.date] = self.index
        self[self.date] = self[self.date].apply(lambda x: int(floor(x/100)))
        self = self.set_index(self.date)
        return self
    def merge(self,A,suffixes=False) :
        if suffixes :
            A = pd.merge(self,A,left_index=True,right_index=True,how='outer',suffixes=suffixes)
        else :
            A = pd.merge(self,A,left_index=True,right_index=True,how='outer')
        return ReturnTable(A,date=self.date)
    def alpha(self,returns,indep) :
        if not(isinstance(returns,list)) : returns = [returns]
        X = DataFrame(self[indep])
        X['Alpha'] = 1.0
        D = DataFrame(columns = ['Alpha', 't-Stat', 'ConfInt_Low', 'ConfInt_High','InfoRatio'],index=returns)
        for ret in returns :
            Y = self[ret]
            result = sm.OLS(Y,X,missing='drop').fit()
            resid_se = sqrt(result.mse_resid)
            alpha = result.params['Alpha']
            info = alpha / resid_se
            conf_lower = result.conf_int()[0][-1]
            conf_upper = result.conf_int()[1][-1]
            D.ix[ret,'Alpha'] = alpha
            D.ix[ret,'t-Stat'] = result.tvalues['Alpha']
            D.ix[ret,'InfoRatio'] = info
            D.ix[ret,'ConfInt_Low'] = conf_lower
            D.ix[ret,'ConfInt_High'] = conf_upper
        return D
    def regr(self,returns,indep) :
        if not(isinstance(returns,list)) : returns = [returns]
        X = DataFrame(self[indep])
        X['Alpha'] = 1.0
        L = []
        for ret in returns :
            Y = self[ret]
            result = sm.OLS(Y,X,missing='drop').fit()
            L.append(result.summary())
        return L
    def cov(self,cols=False,max_periods=False,decay=False,shrink=False,AR=False) :
        if cols : 
           if not(isinstance(cols,list)) : cols = [cols]
           X = DataFrame(self[cols])
        else :
           X = DataFrame(self)
           cols = list(self.columns)
        if max_periods :
            X = X[-max_periods:]
        if AR :
            R = DataFrame(index=self.index,columns=cols)
            for col in cols :
                A = X[col]
                m = tsa.AR(array(A))
                f = m.fit(1)
                p = f.params
                R[col] = A - p[0] - p[1] * A.shift(1)
            R = R[1:]
            if decay :
                if (decay<=0) or (decay>=1) :
                    print 'Warning: The decay parameter is not between 0 and 1.'
                n = R.shape[0]
                vec = array(R[0:1])
                cov = vec.T.dot(vec)
                for i in arange(1,n) :
                    vec = array(R[i:i+1])
                    cov = decay*cov + (1-decay)*vec.T.dot(vec)
                cov = DataFrame(cov,index=cols,columns=cols)
            else :
                cov = R.cov()
        elif decay :
            if (decay<=0) or (decay>=1) :
                print 'Warning: The decay parameter is not between 0 and 1.'
            n = X.shape[0]
            vec = array(X[0:1])
            cov = vec.T.dot(vec)
            for i in arange(1,n) :
                vec = array(X[i:i+1])
                cov = decay*cov + (1-decay)*vec.T.dot(vec)
            cov = DataFrame(cov,index=cols,columns=cols)
        else :
            if len(cols)==1 : cov = var(array(X))
            else : cov = X.cov()
        if shrink :
            if (shrink<=0) or (shrink>=1) :
                print 'Warning: The shrinkage parameter is not between 0 and 1.'
            cov = ShrinkCovs(cov,delta=shrink)
        return DataFrame(cov,index=X.columns,columns=X.columns)
    def corr(self,cols=False,max_periods=False,decay=False,shrink=False,AR=False) :
        cov = self.cov(cols,max_periods,decay,shrink,AR) 
        x = sqrt(diag(cov))
        corrs = ((cov/x).T / x).T
        return DataFrame(corrs,index=cov.index,columns=cov.columns)
    def std(self,cols=False,max_periods=False,decay=False,shrink=False,AR=False) :
        cov = self.cov(cols,max_periods,decay,shrink,AR) 
        x = sqrt(diag(cov))
        return Series(x,index=cov.index)
    def mean(self,cols=False,max_periods=False,decay=False,shrink=False,AR=False,targetzero=False) :
        if cols : 
            if not(isinstance(cols,list)) : cols = [cols]
            X = DataFrame(self[cols])
        else :
            X = DataFrame(self)
            cols = list(self.columns)
        if max_periods :
            X = X[-max_periods:]
        if AR :
            mn = Series(index=cols)
            for col in cols :
                A = self[col]
                m = tsa.AR(array(A))
                f = m.fit(1)
                p = f.params
                mn[col] = p[0] + p[1]*A[-1:]
        elif decay :
            if (decay<=0) or (decay>=1) :
                print 'Warning: The decay parameter is not between 0 and 1.'
            a = X[0]
            for i in arange(1,n) :
                a = decay*a + (1-decay)*X[i]
            mn = Series(a,index=cols)
        else :
            mn = X.mean()
        if shrink :
            if targetzero :
                mn = (1-shrink)*mn 
            else :
                mn = (1-shrink)*mn + shrink*mean(mn)
        return mn
    def tvals(self,cols=False,max_periods=False) :
        if not(cols) : cols = list(self.columns) 
        if max_periods : 
            X = self[cols][-max_periods:]
        else :
            X = self[cols]
        n = X.count()
        m = X.mean()
        s = X.std()
        return sqrt(n) * m / s
    def autoregr(self,cols=False,degree=1,conf=0.95) :
        if cols : 
            if not(isinstance(cols,list)) : cols = [cols]
        else :
            cols = list(self.columns)
        names = ['b'+ repr(i) for i in range(degree+1)]
        names[0] = 'a'
        names2 = []
        for name in names :
            names2.append(name + '_lower')
            names2.append(name + '_upper')
        P = DataFrame(index=names,columns=cols)
        C = DataFrame(index = names2, columns = cols)
        for col in cols :
            x = array(self[col])
            m = tsa.AR(x)
            f = m.fit(degree)
            c = f.conf_int(alpha=1-conf)
            p = f.params
            C[col] = c.reshape((2*(degree+1),1))
            P[col] = p.reshape((degree+1,1))
        return P, C
    def sharpe(self,cols=False) :
        return self.mean(cols=cols) / self.std(cols=cols)
    def compound(self,cols=False,max_periods=False) :
        if cols : 
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        X = Series(index=cols)
        for col in cols :
            if max_periods: X[col] = (1+self[col][-max_periods:]).prod() - 1
            else : X[col] = (1+self[col]).prod() - 1
        return X
    def geomean(self,cols=False,max_periods=False) :
        if cols : 
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        X = Series(index=cols)
        for col in cols :
            if max_periods: 
                arr = self[col][-max_periods:]
                X[col] = (1+arr).prod()**(1/len(arr)) - 1
            else : X[col] = (1+self[col]).prod()**(1/len(self[col])) - 1
        return X
    def skew(self,cols=False,max_periods=False) :
        if cols : 
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        if max_periods: 
            return self[cols][-max_periods:].skew()
        else : 
            return self[cols].skew()
    def kurt(self,cols=False,max_periods=False) :
        if cols : 
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        if max_periods: 
            return self[cols][-max_periods:].kurt()
        else : 
            return self[cols].kurt()
    def select(self,items=False,first=False,last=False) :
        if items :
            if not(isinstance(items,list)) :
                items = [items]
            List = []
            A = DataFrame(self)
            cols = A.columns
            n = len(cols)
            for item in items :
                X = A.ix[item]
                X = DataFrame(X.reshape((1,n)),index=[item],columns=cols)
                List.append(X)
            X = pd.concat(List)
            return ReturnTable(X,date=self.date)
        elif bool(first) & last :
            X = DataFrame(self)
            X = X.ix[first:last]
            return ReturnTable(X,date=self.date)
        elif first :
            X = DataFrame(self)
            X = X.ix[first:]
            return ReturnTable(X,date=self.date)
        else :
            X = DataFrame(self)
            X = X.ix[:last]
            return ReturnTable(X,date=self.date)
'''

END OF RETURN TABLE / START STOCK TABLE FUNCTIONS

'''

def groups(List,cols,List_numvals) :
    if len(cols) == 0 :
        return List
    elif len(List) == 0 :
        List = [cols[0] + '_' + str(a) for a in List_numvals[0]]
        return groups(List,cols[1:],List_numvals[1:])
    else :
        List2 = []
        for x in List :
            for a in List_numvals[0] :
                List2.append(x + '_' + '&' +  '_' + cols[0] + '_' + str(a))
        return groups(List2,cols[1:],List_numvals[1:])
        
def weightedsum(df,char,weightby) :
    return sum(df[char] * df[weightby]) / sum(df[weightby])

  
def cs_winFn(ser,level) :
    indx = ser.index
    ser = ser.dropna()
    arr = array(ser)
    arr = winsorize(arr,limits=(level,level))
    ser = Series(arr,index=ser.index)
    return ser.reindex(indx)
   
def cs_rescaleFn(arr,num,how) :
    assert how in ['std','sum','sumabs'], " 'how' is not defined correctly in cs_rescale"
    if how == 'std' :
        return num * arr / arr.std()
    elif how == 'sum' :
        return num * arr / arr.sum()
    else :
        return num * arr / abs(arr).sum()
    
def cs_regrFn(df,dep,indep,weightby):
    if weightby :
        df = df.dropna(subset=indep+[dep,weightby],how='any')
    else :
        df = df.dropna(subset=indep+[dep],how='any')
    X = df[indep]
    X['Intercept'] = 1.0
    if weightby :
        D = diag(df[weightby]**2)
        RHS = X.T.dot(D)
    else :
        RHS = X.T
    LHS = dot(RHS,X)
    Ptranspose = linalg.solve(LHS,RHS)
    P = Ptranspose.T
    return DataFrame(P,index=df.index,columns = indep+['Intercept'])
        
def cs_binFn_u(df,cols,nums,weightby=False,extreme=False) :
    Cols = []
    for col, num in zip(cols,nums) :
        df2 = df.dropna(subset=[col],how='any')
        num_vals = arange(1,num+1)
        if (extreme == 'on') or (extreme == 'hml') or (extreme == 'lmh'):
            num_vals = [1,num]
        else :
            num_vals = arange(1,num+1,1)
        if df[col].count() < num :
            df2['Group'] = nan
        elif (extreme == 'on') or (extreme == 'hml') or (extreme == 'lmh'):
            factors = pd.qcut(df2[col],[0,1.0/num,1-1.0/num,1])
            grouped = df2.groupby(factors)
            df2['Group'] = [num if x==3 else x for x in factors.labels + 1]
        else :
            factors = pd.qcut(df2[col],num)
            grouped = df2.groupby(factors)
            df2['Group'] = factors.labels + 1
        for num in num_vals :
            name = col + '_' + str(num)
            Cols.append(name)
            df2[name] = (df2['Group'] == num)
            if weightby: df2[name] = df2[weightby] * df2[name]
            total = df2[name].sum()
            if total != 0 : df2[name] = df2[name] / total
            df[name] = df2[name] 
    if extreme == 'hml' :
        Cols = []
        for col, num in zip(cols,nums) :
            df[col+'_'+str(num)+'-'+str(1)] = df[col+'_'+str(num)] - df[col+'_'+str(1)]
            Cols.append(col+'_'+str(num)+'-'+str(1))
    elif extreme == 'lmh' :
        Cols = []
        for col, num in zip(cols,nums) :
            df[col+'_'+str(1)+'-'+str(num)] = df[col+'_'+str(1)] - df[col+'_'+str(num)]
            Cols.append(col+'_'+str(1)+'-'+str(num))
    return DataFrame(df[Cols])    

def cs_binFn_m(df,cols,nums,weightby=False,extreme=False) :
    df2 = df.dropna(subset=cols,how='any')
    List = []
    List_numvals=[]
    for col, num in zip(cols,nums) :
        if df2[col].count() < num :
            df2['Group'] = nan
        elif extreme == 'on' :
            numvals = [1,num]
            group_vals = [1,3]
            factors = pd.qcut(df2[col],[0,1.0/num,1-1.0/num,1])
            labs = array([num if x==3 else x for x in factors.labels+1])
        else :
            numvals = arange(1,num+1)
            group_vals = numvals
            factors = pd.qcut(df2[col],num)
            labs = factors.labels + 1
        List.append(labs)
        List_numvals.append(numvals)
    Cols = groups([],cols,List_numvals)
    labs = [cols[0] + '_' + str(x) for x in List[0]]
    for i, g in enumerate(List[1:]) :
       labs = [x + '_' + '&' + '_' + cols[i+1] + '_' + str(y) for x, y in zip(labs,g)]
    df2['Group'] = labs
    for name in Cols: 
        df2[name] = 1.0*(df2['Group']==name)
        if weightby: df2[name] = df2[weightby] * df2[name]
        total = sum(df2[name])
        if total != 0 :
            df2[name] = df2[name] / total
        df[name] = df2[name]
    return DataFrame(df[Cols])   

def ts_countFn(arr,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_count(arr,max_periods)
    
def ts_sumFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_sum(arr,max_periods,min_periods=min_periods)
    
def ts_meanFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_mean(arr,max_periods,min_periods=min_periods)

def ts_medianFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_median(arr,max_periods,min_periods=min_periods)
    
def ts_stdFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_std(arr,max_periods,min_periods=min_periods)

def ts_varFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_var(arr,max_periods,min_periods=min_periods)    

def ts_skewFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_skew(arr,max_periods,min_periods=min_periods) 
    
def ts_kurtFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_kurt(arr,max_periods,min_periods=min_periods) 
    
def ts_minFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_min(arr,max_periods,min_periods=min_periods)
    
def ts_maxFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_max(arr,max_periods,min_periods=min_periods)
    
def ts_quantileFn(arr,q,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_quantile(arr,max_periods,min_periods=min_periods,quantile=q) 
    
def ts_compoundFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_apply(arr,max_periods,lambda arr: (1+arr).prod()-1,min_periods=min_periods) 

def ts_geomeanFn(arr,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(arr)
    return pd.rolling_apply(arr,max_periods,lambda arr: (1+arr).prod()**(1/len(arr))-1,min_periods=min_periods)

def ts_corrFn(df,col1,col2,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(df[col1])
    return pd.rolling_corr(df[col1],df[col2],max_periods,min_periods=min_periods)
    
def ts_covFn(df,col1,col2,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(df[col1])
    return pd.rolling_cov(df[col1],df[col2],max_periods,min_periods=min_periods)    

def ts_regrFn(df,dep,indep,min_periods,max_periods) :
    if not(max_periods) : max_periods = len(df[dep])
    indx = df.index
    names = indx.names
    cols = [col+'_beta' for col in indep] + ['intercept']
    df = df.reset_index([0])
    X = df[indep+[dep]].dropna(how='any')
    if min(X.count()) >= min_periods :
        model = pd.ols(y=df[dep],x=df[indep],window_type='rolling',window=max_periods,min_periods=min_periods)
        X =  model.beta
        X = pd.merge(df,X,left_index=True,right_index=True,how='outer',suffixes=['','_beta'])
        X = X.reset_index()
        X = X.set_index(names)
        return X[cols]
    else :
        return DataFrame(nan,index=indx,columns=cols)
        
def largestFn(df,char,num) :
    x = df[char].rank(ascending=False)
    return df[x < num+1]

def smallestFn(df,char,num) :
    x = df[char].rank()
    return df[x < num+1]

SIC_Codes = {1:(100,999),2:(1000,1499),3:(1500,1799),4:(2000,3999),5:(4000,4999),6:(5000,5199),7:(5200,5999),8:(6000,6799),9:(7000,8999)}
def SIC(x) :
    for key in SIC_Codes :
        if SIC_Codes[key][0] <= x <= SIC_Codes[key][1] :
            return key
    return nan
    
'''

START STOCK TABLE

'''
          
class StockTable(DataFrame) :
    def __init__(self,df,date='d',stock='s') :
        super(StockTable,self).__init__(df)
        self.date = date
        self.stock = stock
        self.sort_index(inplace=True)
    def __repr__(self) :
        return DataFrame.__repr__(self)
    def __str__(self) :
        return DataFrame.__str__(self)
    def SIC(self,col,name) :
        X = self[col].replace('Z',nan)
        X = X.dropna()
        X = X.map(int)
        X = X.map(SIC)
        self[name] = X
    def changedates(self,old,new,newtype='int') :
        R = DataFrame(self).reset_index()
        ch = vectorize(lambda x: convert(x,old,new,newtype))
        R[self.date] = ch(array(R[self.date]))
        _ = R.set_index([self.date,self.stock],drop=True,inplace=True)
        return StockTable(R,date=self.date,stock=self.stock)
    def swapindex(self) :
        X = DataFrame(self)
        X = X.swaplevel(0,1).sortlevel(0)
        return StockTable(X,date=self.date,stock=self.stock)
    def head(self) :
        X = DataFrame(self)
        return X.head()
    def tail(self) :
        X = DataFrame(self)
        return X.tail()
    def cs_weight(self,col,name) :
        grouped = self[col].groupby(level=self.date)
        self[name] = grouped.apply(lambda arr: arr / arr.sum())
    def cs_count(self, cols=False,zeros=True) :
        if not(cols) : cols = list(self.columns)
        if zeros :
            grouped = self[cols].groupby(level=self.date)
        else :
            X = self[cols].replace(0,nan)
            grouped = X.groupby(level=self.date)
        return ReturnTable(grouped.count(),date=self.date)
    def cs_sum(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.sum(),date=self.date)
    def cs_mean(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.mean(),date=self.date)
    def cs_median(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.median(),date=self.date)
    def cs_std(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.std(),date=self.date)
    def cs_var(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.var(),date=self.date)
    def cs_skew(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.skew(),date=self.date)
    def cs_kurt(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.kurt(),date=self.date)
    def cs_min(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.min(),date=self.date)
    def cs_max(self, cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.max(),date=self.date)
    def cs_quantile(self,q, cols=False,) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        return ReturnTable(grouped.quantile(q),date=self.date)
    def cs_corr(self,cols=False) :
        if not(cols) : cols = list(self.columns)
        grouped = self[cols].groupby(level=self.date)
        Names = []
        Corrs = []
        for i, col1 in enumerate(cols):
            for col2 in cols[(i+1):] :
                X = grouped.apply(lambda g: g[col1].corr(g[col2]))
                Names.append(col1+'_'+col2)
                Corrs.append(X)
        X = pd.concat(Corrs,keys=Names,axis=1)
        return ReturnTable(X,date=self.date)
    def cs_sumprod(self,cols=False,multby=False) :
        if cols :
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        grouped = self.groupby(level=self.date)
        List = []
        for col in cols :
            X = grouped.apply(lambda g: sum(g[col]*g[multby]))
            List.append(X)
        X = pd.concat(List,keys=cols,axis=1)
        return ReturnTable(X,date=self.date)
    def cs_wtavg(self,cols=False,weightby=False) :
        if cols :
            if not(isinstance(cols,list)) : cols=[cols]
        else : 
            cols = list(self.columns)
        grouped = self.groupby(level=self.date)
        List = []
        for col in cols :
            X = grouped.apply(lambda g: average(g[col],weights=g[weightby]))
            List.append(X)
        X = pd.concat(List,keys=cols,axis=1)
        return ReturnTable(X,date=self.date)
    def cs_win(self,cols,level=0.01) :
        if not(isinstance(cols,list)) : cols = [cols]
        for col in cols :
            self[col] = self[col].groupby(level=self.date).transform(lambda arr: cs_winFn(arr,level))
    def cs_demean(self,cols) :
        if not(isinstance(cols,list)) : cols = [cols]
        for col in cols :
            self[col] = self[col].groupby(level=self.date).transform(lambda arr: arr-arr.mean())
    def cs_rescale(self,cols,num,how) :
        if not(isinstance(cols,list)) : cols = [cols]
        for col in cols :
            self[col] = self[col].groupby(level=self.date).transform(cs_rescaleFn,num,how)
        return self
    def cs_regr(self,dep,indep,weightby=False,resid=False,name=False) :
        if not(isinstance(indep,list)) : indep = [indep]
        cols = indep + ['Intercept']
        all_cols = list(self.columns)
        grouped = self.groupby(level=self.date)
        P = grouped.apply(cs_regrFn,dep,indep,weightby)
        _ = P.reset_index([1],drop=True,inplace=True)
        P = StockTable(P,date=self.date,stock=self.stock)
        P[dep] = self[dep]
        R = P.cs_sumprod(cols,multby=dep)
        P = P[cols]
        P = StockTable(P,date=self.date,stock=self.stock)
        if resid :
            if not(name) : name = 'Resid'
            A = self.merge(R,Return=True)
            self[name] = A[dep] - A['Intercept']
            for col in indep :
                self[name] -= A[col+'_x'] * A[col+'_y']
        return P, R
    def cs_bin(self,cols,nums,val=False,weightby=False,intersect=False,extreme=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if not(isinstance(nums,list)) : nums = [nums] * len(cols)
        if extreme :
            assert extreme=='on' or extreme=='hml' or extreme=='lmh', "if extreme is specified in cs_bin, it must on or hml or lmh."
        if intersect and (extreme=='hml' or extreme=='lmh'):
            print "\nWarning: intersect=True is specified in cs_bin, so extreme='hml' or 'extreme=lmh' is being replaced by extreme='on'\n"
            extreme = 'on'
        grouped = self.groupby(level=self.date)
        if intersect :
            P = grouped.apply(cs_binFn_m,cols,nums,weightby,extreme)
        else :
            P = grouped.apply(cs_binFn_u,cols,nums,weightby,extreme)
        P = StockTable(P,date=self.date,stock=self.stock)
        N = P.cs_count(zeros=False)
        P = P.fillna(0)
        P = StockTable(P,date=self.date,stock=self.stock)
        if val :
            A = P.join(self[val])
            A = StockTable(A,date=self.date,stock=self.stock)
            R = A.cs_sumprod(list(P.columns),multby=val)
            return P, N, R
        else :
            return P, N
    def ts_count(self,cols,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_count' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_countFn,min_periods,max_periods)
    def ts_sum(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_sum' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_sumFn,min_periods,max_periods)
    def ts_mean(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_mean' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_meanFn,min_periods,max_periods)
    def ts_median(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_median' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_medianFn,min_periods,max_periods)
    def ts_std(self,cols,min_periods=2,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_std' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_stdFn,min_periods,max_periods)
    def ts_var(self,cols,min_periods=2,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_var' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_varFn,min_periods,max_periods)
    def ts_skew(self,cols,min_periods=2,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_skew' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_skewFn,min_periods,max_periods)
    def ts_kurt(self,cols,min_periods=2,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_kurt' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_kurtFn,min_periods,max_periods)
    def ts_min(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_min' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_minFn,min_periods,max_periods)
    def ts_max(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_max' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_maxFn,min_periods,max_periods)
    def ts_quantile(self,cols,qlist,min_periods=2,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if not(isinstance(qlist,list)) : qlist = [qlist]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_quantile' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name, q in zip(cols,names,qlist) :
            self[name] = grouped[col].transform(ts_quantileFn,q,min_periods,max_periods)
    def ts_compound(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_compound' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_compoundFn,min_periods,max_periods)
    def ts_geomean(self,cols,min_periods=1,max_periods=False,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_geomean' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(ts_geomeanFn,min_periods,max_periods)
    def ts_corr(self,cols,min_periods=2,max_periods=False) :
        grouped = self[cols].groupby(level=self.stock,group_keys=False)
        List=[]
        for i, col1 in enumerate(cols) :
            for col2 in cols[(i+1):] :
               self[col1+'_'+col2+'_corr'] = grouped.apply(ts_corrFn,col1,col2,min_periods,max_periods)
    def ts_cov(self,cols,min_periods=2,max_periods=False) :
        grouped = self[cols].groupby(level=self.stock,group_keys=False)
        List=[]
        for i, col1 in enumerate(cols) :
            for col2 in cols[(i+1):] :
               self[col1+'_'+col2+'_cov'] = grouped.apply(ts_covFn,col1,col2,min_periods,max_periods)
    def ts_lag(self,cols=False,periods=1,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_lag' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(lambda arr: arr.shift(periods))
    def ts_change(self,cols=False,periods=1,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_change' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(lambda arr: arr - arr.shift(periods))
    def ts_pchange(self,cols=False,periods=1,names=False) :
        if not(isinstance(cols,list)) : cols = [cols]
        if names :
            if not(isinstance(names,list)) : names = [names]
        else :
            names = [col+'_pchange' for col in cols]
        grouped = self[cols].groupby(level=self.stock)
        for col, name in zip(cols,names) :
            self[name] = grouped[col].transform(lambda arr: arr/arr.shift(periods) -1)
    def ts_regr(self,dep,indep,min_periods=1,max_periods=False) :
        if not(isinstance(indep,list)) : indep = [indep]
        min_periods = max(min_periods,len(indep)+1)
        List = []
        df = self.swapindex()
        for stock in list(df.index.levels[0]) :
            A = df.select(stock)[indep+[dep]]
            A = ts_regrFn(A,dep,indep,min_periods,max_periods) 
            List.append(A)
        A = pd.concat(List)
        A = A.swaplevel(0,1)
        return StockTable(A,date=self.date,stock=self.stock)
    def merge(self,A,Return=False,suffixes=False,how='outer') :
        if Return :  A = A.reindex(self.index,level=self.date)
        if suffixes :
            A = pd.merge(self,A,left_index=True,right_index=True,suffixes=suffixes,how=how)
        else :
            A = pd.merge(self,A,left_index=True,right_index=True,how=how)
        
        A.index.names = [self.date,self.stock]
        return StockTable(A,date=self.date,stock=self.stock)
    def write(self, filename) :
        self.to_csv(filename+'.csv', index=True)
    def wide(self,var) :
        X = self.unstack(self.stock)[var]
        return ReturnTable(X,date=self.date)
    def dropday(self) :
        x = self.index
        date = self.date
        stock = self.stock
        self[date] = [a[0] for a in x]
        self[stock] = [a[1] for a in x]
        self[date] = self[date].apply(lambda x: int(int(x)/100))
        self = self.set_index([date,stock],drop=True)
        return StockTable(self,date=date,stock=stock)
    def dropdash(self) :
        x = self.index
        date = self.date
        stock = self.stock
        self[date] = [a[0] for a in x]
        self[stock] = [a[1] for a in x]
        self[date] = self[date].apply(lambda x: int(x.replace('-','')))
        self = self.set_index([date,stock],drop=True)
        return StockTable(self,date=date,stock=stock)
    def select(self,items=False,first=False,last=False,stock=False,dropdate=False) :
        if stock : 
            A = DataFrame(self.swapindex())
        else :
            A = DataFrame(self)
        if items :
            if not(isinstance(items,list)) :
                items = [items]
            assert dropdate is False or len(items)==1, "you cannot use dropdate if you are selecting multiple dates"
            List = []
            outer = A.index.names[0]
            inner = A.index.names[1]
            for item in items :
                X = A.ix[item]
                X['new_outer_index'] = item
                X['new_inner_index'] = array(X.index)
                X = X.set_index(['new_outer_index','new_inner_index'])
                X.index.names = [outer,inner]
                List.append(X)
            A = pd.concat(List)
            A = StockTable(A,date=self.date,stock=self.stock)
            if dropdate :
                A = DataFrame(A).reset_index()
                del A[self.date]
                A = A.set_index(self.stock,drop=True)
        elif bool(first) & last :
            A = StockTable(A.ix[first:last],date=self.date,stock=self.stock)
        elif first :
            A = StockTable(A.ix[first:],date=self.date,stock=self.stock)
        else :
            A = StockTable(A.ix[:last],date=self.date,stock=self.stock)
        if stock :
            return A.swapindex()
        else :
            return A
    def subset(self,col,lst) :
        if not(isinstance(lst,list)) : lst = [lst]
        List = []
        for item in lst :
            X = (self[col] == item)
            List.append(X)
        X = List[0]
        for Y in List[1:] :
            X = any((X,Y),axis=0)
        return StockTable(self[X],date=self.date,stock=self.stock)
    def largest(self,char,num) :
        grouped = self.groupby(level=self.date)
        X = grouped.apply(largestFn,char,num)
        X = X.reset_index([1],drop=True)
        return StockTable(X,date=self.date,stock=self.stock)
    def smallest(self,char,num) :
        grouped = self.groupby(level=self.date)
        X = grouped.apply(smallestFn,char,num)
        X = X.reset_index([1],drop=True)
        return StockTable(X,date=self.date,stock=self.stock)

   
'''

START MEAN VARIANCE

'''
       
class MeanVar() :
    def __init__(self,means,covs,names=False) :
        if names :
            self.names = names
        elif isinstance(means,Series) :
            self.names = list(means.index)
        elif isinstance(covs,DataFrame) :
            self.names = list(covs.index)
        else :
            self.names = range(1,len(array(means))+1)
        self.mns = array(means)
        self.cvs = array(covs)
        self.number = len(self.names)
    def means(self) :
        return Series(self.mns,index=self.names)
    def covs(self) :
        return DataFrame(self.cvs,index=self.names,columns=self.names)
    def __repr__(self) :
        X = DataFrame(self.means(),columns=['Means'])
        X = pd.merge(X,self.covs(),left_index=True,right_index=True,how='outer')
        return DataFrame.__repr__(X)
    def __str__(self) :
        X = DataFrame(self.means(),columns=['Means'])
        X = pd.merge(X,self.covs(),left_index=True,right_index=True,how='outer')
        return DataFrame.__str__(X)
    def head(self) :
        X = self.df()
        return X.head()
    def tail(self) :
        X = self.df()
        return X.tail()
    def mean(self,w) :
        return self.mns.dot(array(w)).item() 
    def var(self,w) :
        return array(w).T.dot(self.cvs).dot(array(w)).item()
    def std(self,w) :
        return sqrt(self.var(w))
    def sharpe(self,w,rf=0) :
        m = self.mean(w) - rf
        s = self.std(w)
        return m / s
    def sumweights(self,w,ports=False) :
        if ports :
            return sum(array(ports).dot(array(w)))
        else :
            return sum(array(w))
    def sumabsweights(self,w,ports=False) :
        if ports :
            return sum(abs(array(ports).dot(array(w))))
        else :
            return sum(abs(array(w)))
    def feq(self,w,target,sumwts,ports) :
        x = self.mean(w) - target
        if not(sumwts is False) :
            y = self.sumweights(w,ports) - sumwts
            return hstack((x,y))
        else :
            return array([x])
    def meq(self,w,sumwts,ports) :
        return array([self.sumweights(w,ports) - sumwts])
    def ieqs(self,w,minwt,maxwt,sumabswts,ports) :
        if ports :
            x = array(ports).dot(array(w))
        else :
            x = w
        if not(sumabswts is False) :
            y = array([sumabswts - self.sumabsweights(w,ports)])
            if not(minwt is False) and not(maxwt is False) :
                return hstack((maxwt - x,x - minwt,y))
            elif not(minwt is False) :
                return hstack((x-minwt,y))
            elif not(maxwt is False) :
                return hstack((maxwt-x,y))
            else :
                return y
        else :
            if not(minwt is False) and not(maxwt is False) :
                return hstack((maxwt - x,x - minwt))
            elif not(minwt is False) :
                return x - minwt
            else :
                return maxwt - x
    def frontier(self,target,sumwts=False,sumabswts=False,minwt=False,maxwt=False,ports=False,startpt=False, \
        iter=100, acc=1e-06, iprint=0, disp=None, full_output=0, epsilon=1.4901161193847656e-08) :
        if startpt is False : startpt = ones(self.number)/float(self.number)
        if sumwts and ports :
            x = ports.sum().abs().sum()
            if x < 1.0e-3 : print "\nWarning: It appears you are trying to force long-short portfolios to sum to something other than zero.\n"
        if not(minwt is False) or not(maxwt is False) or not(sumabswts is False) : 
            res = fmin_slsqp(self.std,startpt,f_eqcons=(lambda w: self.feq(w,target,sumwts,ports)),f_ieqcons=(lambda w: self.ieqs(w,minwt,maxwt,sumabswts,ports)) ,iprint=0)
        else : 
            res = fmin_slsqp(self.std,startpt,f_eqcons=(lambda w: self.feq(w,target,sumwts,ports)),f_ieqcons=None ,iprint=0)
        p = DataFrame(res,index=self.names,columns=['Weight'])
        if ports : 
            return p, ports.dot(p)
        else : 
            return p
    def utility(self,penalty,w) :
        return self.mean(w) - penalty * self.var(w)
    def optimal(self,penalty,sumwts=False,sumabswts=False,minwt=False,maxwt=False,ports=False,startpt=False, \
         iter=100, acc=1e-06, iprint=0, disp=None, full_output=0, epsilon=1.4901161193847656e-08) :
        if startpt is False : startpt = ones(self.number)/float(self.number)
        if sumwts and ports :
            x = ports.sum().abs().sum()
            if x < 1.0e-3 : print "\nWarning: It appears you are trying to force long-short portfolios to sum to something other than zero.\n"
        ineq_true = not(minwt is False) or not(maxwt is False) or not(sumabswts is False)
        if not(sumwts is False) and ineq_true :
            res = fmin_slsqp(lambda w: - self.utility(penalty,w),startpt,f_eqcons=(lambda w: self.meq(w,sumwts,ports)),f_ieqcons=(lambda w: self.ieqs(w,minwt,maxwt,sumabswts,ports)),iprint=0)
        elif not(sumwts is False) :
            res = fmin_slsqp(lambda w: - self.utility(penalty,w),startpt,f_eqcons=(lambda w: self.meq(w,sumwts,ports)),f_ieqcons=None,iprint=0)
        elif ineq_true :
            res = fmin_slsqp(lambda w: penalty* self.var(w)-self.mean(w),startpt,f_eqcons=None,f_ieqcons=(lambda w: self.ieqs(w,minwt,maxwt,sumabswts,ports)),iprint=0)
        else : 
            res = fmin_slsqp(lambda w: penalty* self.var(w)-self.mean(w),startpt,f_eqcons=None,f_ieqcons=None,iprint=0)
        p = DataFrame(res,index=self.names,columns=['Weight'])
        if ports : 
            return p, ports.dot(p)
        else : 
            return p