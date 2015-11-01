# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:26:36 2014

@author: kerry
"""

means = [0.10,0.08,0.15]
stdevs = array([0.3,0.2,0.4])
corrs = array([[1.0,0.2,0.1],[0.2,1.0,0.5],[0.1,0.5,1.0]])
covs = diag(stdevs).dot(corrs).dot(diag(stdevs))

StockNames = ['Stock1','Stock2','Stock3','Stock4','Stock5']
A = Series([0.2,0.1,0.3,0.1,0.3],index=StockNames)
B = Series([0.1,0.1,0.5,0.1,0.2],index=StockNames)
C = Series([0.3,0.2,0.1,0.2,0.2],index=StockNames)
P = DataFrame(A,columns=['A'])
P['B'] = B
P['C'] = C

StockNames = ['Stock1','Stock2','Stock3','Stock4','Stock5']
P = DataFrame([0.2,0.1,0.3,0.1,0.3],index=StockNames,columns=['A'])
P['B'] = [0.1,0.1,0.5,0.1,0.2]
P['C'] = [0.3,0.2,0.1,0.2,0.2]


runfile('/home/kerry/Dropbox/Quantmodels USU 2014/quantmodels.py')

X = MeanVar(means,covs,names=['A','B','C'])




p = [0.2,0.5,0.3]
X.mean(p)
X.var(p)
X.std(p)
X.sharpe(p,rf=0.02)

p1 = X.frontier(0.12)
p2 = X.optimal(2)
p3 = X.frontier(0.12,sumwts=1)
p4 = X.optimal(2,sumwts=0)
p5 = X.frontier(0.12,sumwts=1,minwt=0)/
p6 = X.optimal(2,sumwts=1,maxwt=[0.4,.2,0.5])

X.std(p1)
X.std(p3)
X.std(p5)
X.mean(p1)
X.mean(p3)
X.mean(p5)

X.utility(2,p2)
X.utility(2,p4)
X.utility(2,p6)

p1= X.frontier(0.12,sumwts=1,sumabswts=2)
p2 = X.frontier(0.12,sumwts=1)
p3 = X.frontier(0.12,sumabswts=2)

X.feq(p1,0.12,1,False)
X.feq(p2,0.12,1,False)
X.feq(p3,0.12,False,False)

X.ieqs(p1,False,False,2,False)

p1.sum()
p1.abs().sum()
p2.sum()
p3.abs().sum()


p1, stocks1 = X.frontier(0.12,ports=P)
p2, stocks2 = X.optimal(2,ports=P)

p1 = X.frontier(0.12,sumwts=1,minwt=0,maxwt=0.5,ports=P) 
p2 = X.frontier(0.12,sumwts=1,minwt=0,maxwt=0.5,ports=P,startpt=[0.2,0.5,0.3]) 