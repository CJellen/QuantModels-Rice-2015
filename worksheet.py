workingdir = '/home/kerry/Dropbox/Quantmodels USU 2014/'
import os
os.chdir(workingdir)
from quantmodels import *

S = stock_read('StockData',date='DATE',stock='PERMNO')

chars = ['AG', 'BM', 'CEI', 'CHSDP', 'GP', 'IA', 'IVOL', 'NOA', 'NSI', 'OSCORE', 'RET_2_7', 'RET_2_12', 'TME',]
W, N, R = S.cs_bin(chars,10,weightby='ME',val='RET',extreme='hml')
months = list(R.index) 
past = [m for m in months if m < 198500]
future = [m for m in months if m > 198500]
rets = DataFrame(index=future,columns=['RET'])

for month in future :
    
    Rpast = R.select(last=past[-1])
    means = Rpast.mean(max_periods=120,shrink=1/3)
    covs = Rpast.cov(max_periods=120,shrink=1/3)  
    M = MVO(W.select(month),means,covs,maxwt=0.05)
    
    P, w = M.maxutility(4)
    rets.ix[month] = array(R.select(month)).dot(array(w))
    past.append(month)
    
for month in future :
    if month % 100 in [1,4,7,10] :
        Rpast = R.select(last=past[-1])
        means = Rpast.mean(max_periods=120)
        covs = Rpast.cov(max_periods=120)  
        M = MVO(W.select(month),means,covs)
        P, w = M.frontier(0.01)
    rets.ix[month] = array(R.select(month)).dot(array(w))
    past.append(month)

rets.index.name = 'DATE'
rets = ReturnTable(rets,date='DATE')
sqrt(12)*rets.sharpe()
    
F = return_read('FrenchData',date='DATE')
F = F.select(first=198501)
F = F.merge(rets)
F.alpha('RET',['Mkt-RF','SMB','HML','UMD'])


S = stock_read('StockData',date='DATE',stock='PERMNO')

chars = ['AG', 'BM', 'CEI','RET_2_12']
W, N, R = S.cs_bin(chars,10,weightby='ME',val='RET',extreme='hml')

months = list(R.index) 
past = [m for m in months if m < 198500]
future = [m for m in months if m > 198500]

rets = DataFrame(index=future,columns=['RET'])
rets.index.name = 'DATE'
rets = ReturnTable(rets,date='DATE')

for month in future :
    PastReturns = R.select(last=past[-1])
    Means = PastReturns.mean(max_periods=120)
    Covs = PastReturns.cov(max_periods=120)  
    X = MeanVar(Means,Covs) 
    P = W.select(month,dropdate=True)
    port, stocks = X.optimal(2,minwt=-0.05,maxwt=0.05,sumabswts=2,ports=P)
    rets.ix[month] = array(R.select(month)).dot(array(port))
    past.append(month)

12*rets.mean()
sqrt(12)*rets.std()
sqrt(12)*rets.sharpe()