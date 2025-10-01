from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import random

base_pipeline = [('scaler',StandardScaler())]
SEED = 297
random.seed(SEED)
np.random.seed(SEED)
models = {
    'DecisionTree': {
        'pipe': Pipeline(base_pipeline + [('model', DecisionTreeClassifier(random_state=SEED))]),
        'params_grid': {
            'model__max_depth': [5, 8, 12, 16, 20, None],
            'model__class_weight': [None, 'balanced'],
        }
    },

    'RandomForest': {
        'pipe': Pipeline(base_pipeline + [('model', RandomForestClassifier(random_state=SEED, n_jobs=8))]),
        'params_grid': {
            'model__n_estimators': [200, 400, 800],
            'model__max_depth': [None, 10, 16, 24],
            'model__class_weight': [None, 'balanced'],
        }
    },

    'GradientBoosting': {
        'pipe': Pipeline(base_pipeline + [('model', GradientBoostingClassifier(random_state=SEED))]),
        'params_grid': {
            # GB tends to like shallow trees + modest lr for 10k rows
            'model__learning_rate': [0.02, 0.05, 0.1],
            'model__n_estimators': [200, 400, 800],
            'model__max_depth': [2, 3, 4, 5],
            'model__class_weight': [None, 'balanced'],  # NOTE: sklearn GB does NOT support class_weight; remove if error.
        }
    },

    'XGB': {
        'pipe': Pipeline(base_pipeline + [('model', XGBClassifier(objective='binary:logistic',
                                                                  eval_metric='logloss',
                                                                  random_state=SEED,
                                                                  scale_pos_weight = 5.8,
                                                                  n_jobs=8))]),
        'params_grid': {
            'model__learning_rate': [0.02, 0.05, 0.1],
            'model__n_estimators': [300, 600, 900],
            'model__max_depth': [3, 5, 7],
        }
    },

    'LGB': {
        'pipe': Pipeline(base_pipeline + [('model', lgb.LGBMClassifier(objective='binary',
                                                                       metric='binary_logloss',
                                                                       random_state=SEED,
                                                                       n_jobs=8))]),
        'params_grid': {
            'model__learning_rate': [0.02, 0.05, 0.1],
            'model__n_estimators': [300, 600, 1000],
            'model__max_depth': [-1, 5, 8, 12],
            'model__is_unbalance': [True, False],
        }
    },

    'CAT': {
        'pipe': Pipeline(base_pipeline + [('model', CatBoostClassifier(
                                                                       random_state=SEED,
                                                                       verbose=False))]),
        'params_grid': {
            #'model__max_depth': [4, 6, 8, 10],
            'model__n_estimators': [1000],
            'model__learning_rate': [0.01124],
            'model__auto_class_weights': ['Balanced'],  
        }
    },
}


# models = {
#     'DecisionTree': {
#         'pipe': Pipeline(base_pipeline + [('model', DecisionTreeClassifier(random_state=SEED))]),
#         'params_grid': {
#             'model__max_depth': [None, 5, 10, 20, 30],
#             'model__min_samples_split': [2, 5, 10, 20],
#             'model__min_samples_leaf': [1, 2, 5, 10],
#             'model__max_features': ['sqrt', 'log2', None],
#             'model__class_weight': ['balanced', None],
#             'model__criterion': ['gini', 'entropy']
#         }
#     },

#     'RandomForest': {
#         'pipe': Pipeline(base_pipeline + [('model', RandomForestClassifier(random_state=SEED, n_jobs=8))]),
#         'params_grid': {
#             'model__n_estimators': [200, 500, 1000],         # Number of trees
#             'model__max_depth': [None, 10, 20, 30],         # Max depth of trees
#             'model__min_samples_split': [2, 5, 10],         # Minimum samples to split a node
#             'model__min_samples_leaf': [1, 2, 4],           # Minimum samples at a leaf node
#             'model__max_features': ['sqrt', 'log2', 0.5],   # Number of features to consider at each split
#             'model__class_weight': ['balanced', 'balanced_subsample']  # Handle imbalance
#         }
#     },

#     'GradientBoosting': {
#         'pipe': Pipeline(base_pipeline + [('model', GradientBoostingClassifier(random_state=SEED))]),
#         'params_grid': {
#             # GB tends to like shallow trees + modest lr for 10k rows
#             'model__learning_rate': [0.02, 0.05, 0.1],
#             'model__n_estimators': [200, 400, 800],
#             'model__max_depth': [2, 3, 4, 5],
#             'model__class_weight': [None, 'balanced'],  # NOTE: sklearn GB does NOT support class_weight; remove if error.
#         }
#     },

#     'XGB': {
#         'pipe': Pipeline(base_pipeline + [('model', XGBClassifier(objective='binary:logistic',
#                                                                   eval_metric='logloss',
#                                                                   random_state=SEED,
#                                                                   scale_pos_weight = 5.8,
#                                                                   n_jobs=8))]),
#         'params_grid': {
#             'model__n_estimators': [100, 300, 500],
#             'model__max_depth': [3, 5, 7, 10],
#             'model__learning_rate': [0.01, 0.05, 0.1],
#             'model__subsample': [0.6, 0.8, 1.0],       # Row sampling for each tree
#             'model__colsample_bytree': [0.6, 0.8, 1.0],# Feature sampling per tree
#             'model__gamma': [0, 0.1, 0.2],             # Minimum loss reduction to make a split
#             'model__reg_alpha': [0, 0.01, 0.1],        # L1 regularization
#             'model__reg_lambda': [1, 1.5, 2],          # L2 regularization
#             #'scale_pos_weight': [1, 5, 10]
#         }
#     },

#     'LGB': {
#         'pipe': Pipeline(base_pipeline + [('model', lgb.LGBMClassifier(objective='binary',
#                                                                        metric='binary_logloss',
#                                                                        random_state=SEED,
#                                                                        n_jobs=8))]),
#         'params_grid': {
#             'model__learning_rate': [0.02, 0.05, 0.1],
#             'model__n_estimators': [300, 600, 1000],
#             'model__max_depth': [-1, 5, 8, 12],
#             'model__is_unbalance': [True, False],
#         }
#     },

#     'CAT': {
#         'pipe': Pipeline(base_pipeline + [('model', CatBoostClassifier(objective='Logloss',
#                                                                        eval_metric='Logloss',
#                                                                        random_state=SEED,
#                                                                        verbose=False))]),
#         'params_grid': {
#             'model__max_depth': [4, 6, 8, 10],
#             'model__n_estimators': [300, 600, 1000],
#             'model__learning_rate': [0.02, 0.05, 0.1],
#             'model__auto_class_weights': [None, 'Balanced', 'SqrtBalanced'],  
#         }
#     },
# }

def load_dataset(test_size):
    df = pd.read_csv('data/datatrain.csv')
    df = df.drop(columns=['User_ID','Created At Year','Created At time'])
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED,stratify=y
    )

    return X_train, X_test, y_train, y_test

def train(model,X_train,Y_train,cv_splits):
    pipeline = models[model]["pipe"]
    if model == "XGB":
        pos = (Y_train == 1).sum()
        neg = (Y_train == 0).sum()
        scale_pos = neg/pos
        pipeline['model'].scale_pos_weight = scale_pos
    param_grid = models[model]["params_grid"]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED) 
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=8,
        refit=True,
    )

    grid_search.fit(X_train,Y_train)

    return grid_search

def eval(grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame):
    best = grid_search.best_estimator_
    y_pred = best.predict(X_test)
    scores = classification_report(y_test,y_pred,labels=[0,1],digits=3,output_dict=True)
    return {
        "estimator": best["model"],
        "preds": y_pred,
        'report': scores
    }

