import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import copy
from scipy.stats import linregress

class MFModel:
    '''
    Base class for Multifidelity inference model.
    '''
    def __init__(self, **args):
        '''
        Init model.
        '''
        return
    
    def fit(self, X_train_lf, y_train_lf, X_train_hf, y_train_hf):
        '''
        Fits a model to low- and high- fidelity samples.
        '''
        raise NotImplementedError
        
    def predict_lf(self, X):
        '''
        Predicts low-fidelity values.
        '''
        raise NotImplementedError

    def predict_hf(self, X):
        '''
        Predicts low-fidelity values.
        '''
        raise NotImplementedError

class MFBaggingRegressorStacked(MFModel):
    
    def __init__(self, **args):
        '''
        Init model.
        '''
        self.model_lf = BaggingRegressor(**copy.deepcopy(args))
        self.model_hf = BaggingRegressor(**copy.deepcopy(args))
    
    def fit(self, X_train_lf, y_train_lf, X_train_hf, y_train_hf):
        '''
        Fits a model to low- and high- fidelity samples using stacking scheme for BaggingRegressor.
        '''
        self.model_lf.fit(X_train_lf, y_train_lf)
        X_train_hf = np.hstack((X_train_hf, self.model_lf.predict(X_train_hf).reshape(-1, 1)))
        self.model_hf.fit(X_train_hf, y_train_hf)
       
    def predict_hf(self, X):
        '''
        Predicts low-fidelity values.
        '''
        y_pred_lf = self.model_lf.predict(X)
        X = np.hstack((X, y_pred_lf.reshape(-1, 1)))

        base_preds = [e.predict(X) for e in self.model_hf.estimators_]

        y_pred_hf = np.mean(base_preds, axis=0)

        rho = linregress(y_pred_lf, y_pred_hf)[0] # get slope

        return rho, y_pred_hf, np.std(base_preds, axis=0)

    def predict_lf(self, X):
        '''
        Predicts low-fidelity values.
        '''
        base_preds = [e.predict(X) for e in self.model_lf.estimators_]

        return np.mean(base_preds, axis=0), np.std(base_preds, axis=0)


def make_mf_predictor(name='gb_stacked'):
    if name=='gb_stacked':
        return MFBaggingRegressorStacked(base_estimator=GradientBoostingRegressor(
                                                            n_estimators=50, 
                                                            max_depth=5,
                                                        ),
                                         n_estimators=20,
                                         max_samples=0.51,
                                         n_jobs=1)
    else:
        raise ValueError("Unknown name, possible options: 'xgb_stacked'")