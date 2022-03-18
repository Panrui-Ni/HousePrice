import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV


train = pd.read_csv("C:/pythonProjecthouse/Data/train_data.csv")
test = pd.read_csv("C:/pythonProjecthouse/Data/test_data.csv")
sample = pd.read_csv("C:/pythonProjecthouse/Data/sample_submission.csv")

y = train['SalePrice']
y = np.log1p(y)
xgbr = xgb.XGBRegressor(learning_rate=0.01, n_estimators=3000, max_depth=5, subsample=0.6, colsample_bytree=0.7, min_child_weight=3, seed=52, gamma=0, reg_alpha=0, reg_lambda=1)
xgbr.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))
#predictions = xgbr.predict(np.array(test))


#rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#rfr.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))
#predictions = rfr.predict(np.array(test))


#gbr = GradientBoostingRegressor(loss='huber', criterion= 'mse', learning_rate=0.1, n_estimators=600, max_depth=4, subsample=0.6, min_samples_leaf=5, min_samples_split=20, max_features=0.6, random_state=32, alpha=0.5)
#gbr.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))

#ridge = make_pipeline(RobustScaler(), RidgeCV())
#ridge.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))

#svr = make_pipeline(RobustScaler(), SVR())
#svr.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))

#stack = StackingCVRegressor(regressors=(xgbr, gbr, ridge, svr, rfr), meta_regressor=ridge, use_features_in_secondary=True)
#stack.fit(np.array(train.drop('SalePrice', axis=1)),np.array(y))
predictions = xgbr.predict(np.array(test))
predictions = np.expm1(predictions)
#print(predictions)
result = pd.DataFrame({'Id': sample['Id'], 'SalePrice': predictions})
result.to_csv("C:/pythonProjecthouse/Data/submission.csv")