from universal.algo import Algo
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVR 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

class ML(Algo):
    # use logarithm of prices
    PRICE_TYPE = 'log'
    
    def __init__(self, n, model='mlp', targettype = 0, predicttype=1, lookbackwindow=10, featurewindow =80, updateFrequency=20):
        # length of moving average
        self.n = n
        self.lookbackwindow = lookbackwindow
        self.featurewindow = featurewindow
        self.updateFrequency = updateFrequency
        self.targettype = targettype
        self.predicttype = predicttype
        self.regr = None
        self.model = model
        # step function will be called after min_history days
        super().__init__(min_history=n)
    
    def init_weights(self, cols):
        # use zero weights for start
        m = len(cols)
        return np.ones(m) / m
    
    def step(self, x, last_b, history):
        # calculate moving average
        nrow = history.shape[0]
        ncol = history.shape[1]
        if history.shape[0] % self.updateFrequency == 0:
            featureMatrix = pd.DataFrame()
            targetMatrix = pd.DataFrame()
            for colidx in range(ncol):
                relativeposition = []
                rankcorrelation = []
                volatility = []
                previousincrease = []
                for i in range(nrow-self.featurewindow, nrow-2):
                    relativeposition.append(np.array(history.iloc[(i-self.lookbackwindow):(i+1),colidx].mean() / history.iloc[(i-self.lookbackwindow):(i+1),colidx].std()) )
                    coef, p = spearmanr(history.iloc[(i-self.lookbackwindow+1):(i+1), colidx],
                                        np.array(range(len(history.iloc[(i-self.lookbackwindow+1):(i+1), colidx]))))
                    rankcorrelation.append(coef)
                    volatility.append(history.iloc[(i-self.lookbackwindow):(i+1),colidx].std())
                    previousincrease.append(history.iloc[-1, colidx])
                
                featureMatrix[str(colidx) + 'relative'] = np.array(relativeposition)
                featureMatrix[str(colidx) + 'rank'] = np.array(rankcorrelation)
                featureMatrix[str(colidx) + 'vol'] = np.array(volatility)
                featureMatrix[str(colidx) + 'preincrease'] = np.array(previousincrease)
            targetMatrix = history.iloc[(nrow-self.featurewindow+2):nrow]
            if self.targettype == 1:
                targetMatrix = targetMatrix.rank(1)
            if self.targettype >= 2:
                targetMatrix = targetMatrix.rank(1)
                targetMatrix = pow(targetMatrix, self.targettype)
                
            # print(featureMatrix.shape)
            # print(targetMatrix.shape)
            if self.model == 'mlp':
                self.regr = MLPRegressor(random_state=10, max_iter=1000,
                                    #early_stopping=True,
                                    hidden_layer_sizes=(20, 20,)
                                    ).fit(featureMatrix, targetMatrix)
            if self.model == 'xgb':
                self.regr = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror',
                                    n_estimators=1000, seed=3)).fit(featureMatrix, targetMatrix)
            if self.model == 'rf':
                    self.regr = RandomForestRegressor(n_estimators=200,
                                                 random_state=0).fit(featureMatrix, targetMatrix)
            if self.model == 'knn':
                self.regr = KNeighborsRegressor(n_neighbors=15,).fit(featureMatrix, targetMatrix)
                
            if self.model == 'svr':
                self.regr = MultiOutputRegressor(SVR(kernel="rbf")).fit(featureMatrix, targetMatrix)
            
            if self.model == 'et':
                self.regr = ExtraTreesRegressor().fit(featureMatrix, targetMatrix)
                
        
        if self.regr is not None:
            featureMatrix = pd.DataFrame() 
            for colidx in range(ncol):
                relativeposition = []
                rankcorrelation = []
                volatility = []
                previousincrease = []
                i = nrow - 1
                relativeposition.append(np.array(history.iloc[(i-self.lookbackwindow):(i+1),colidx].mean() / history.iloc[(i-self.lookbackwindow):(i+1),colidx].std()) )
                coef, p = spearmanr(history.iloc[(i-self.lookbackwindow+1):(i+1), colidx],
                                    np.array(range(len(history.iloc[(i-self.lookbackwindow+1):(i+1), colidx]))))
                rankcorrelation.append(coef)
                volatility.append(history.iloc[(i-self.lookbackwindow):(i+1),colidx].std())
                previousincrease.append(history.iloc[-1, colidx])

                featureMatrix[str(colidx) + 'relative'] = np.array(relativeposition)
                featureMatrix[str(colidx) + 'rank'] = np.array(rankcorrelation)
                featureMatrix[str(colidx) + 'vol'] = np.array(volatility)
                featureMatrix[str(colidx) + 'preincrease'] = np.array(previousincrease)
            w = self.regr.predict(featureMatrix)[0]
            if self.predicttype == 0:
                w[w <0] = 0
            else:
                w = pow( pd.Series(w).rank(), self.predicttype)
            # w[w <0] = 0
            
        else:
            w = 0*x+1
            
        # normalize so that they sum to 1 
        return w / sum(w)