from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random

base_pipeline = [('scaler',StandardScaler())]
SEED = 297
random.seed(SEED)
np.random.seed(SEED)

models = {

    'DecisionTree' : {

        'pipe' : Pipeline(base_pipeline + [('model',DecisionTreeClassifier(random_state=SEED))]),
        'params_grid' : {

            "max_depth" : range(5,50),
            "class_weight" : [None,"balanced"]
        }
    },

    'RandomForest' : {

        'pipe' : Pipeline(base_pipeline + [('model',RandomForestClassifier(random_state=SEED,n_jobs=8))]),
        'params_grid' : {

            "n_estimators": range(50,200), 
            "max_depth" : range(5,50),
            "class_weight" : [None,"balanced"]
        }
    },

    'GradientBoosting' : {

        'pipe' : Pipeline(base_pipeline + [('model',GradientBoostingClassifier(random_state=SEED))]),
        'params_grid' : {

            "learning_rate" : [0.0001,0.0003,0.0005,0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5],
            "n_estimators": range(50,200), 
            "max_depth" : range(5,50),
            "class_weight" : [None,"balanced"]
        }
    },

    'XGB' : {

        'pipe' : Pipeline(base_pipeline + [('model',XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=SEED))]),
        'params_grid' : {

            "learning_rate" : [0.0001,0.0003,0.0005,0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5],
            "n_estimators": range(50,200), 
            "max_depth" : range(5,50),
        }
    },

    'LGB' : {

        'pipe' : Pipeline(base_pipeline + [('model',lgb.LGBMClassifier(objective="binary",metric="binary_logloss",random_state=SEED))]),
        'params_grid' : {

            "learning_rate" : [0.0001,0.0003,0.0005,0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5],
            "n_estimators": range(50,200), 
            "max_depth" : range(5,50),
            "is_unbalance" : [True,False]
        }
    },

    'CAT' : {

        'pipe' : Pipeline(base_pipeline + [('model',CatBoostClassifier(objective="Logloss",eval_metric='Logloss',random_state=SEED))]),
        'params_grid' : {

            "max_depth" : range(5,50),
            "n_estimators": range(50,200),
            "auto_class_weight" : [None,"balanced"],
            "learning_rate" : [0.0001,0.0003,0.0005,0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5],
        }
    },
}

def load_dataset(test_size):
    df = pd.read_csv('data/data.csv')
    df = df.drop(columns=['User_ID','Countries_ID','Created At time','creation_date'])
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED,stratify=y
    )

    return X_train, X_test, y_train, y_test

def train(model,X_train,Y_train,cv_splits):
    pipeline = models[model]["pipe"]
    param_grid = models[model]["param_grid"]
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=SEED) 
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train,Y_train)

    return grid_search

