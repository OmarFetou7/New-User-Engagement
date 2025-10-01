import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,precision_recall_curve,precision_recall_fscore_support
import random

SEED = 297
random.seed(SEED)
np.random.seed(SEED)

def load_dataset():
    data = pd.read_csv("data/datatrain.csv")
    test_df = pd.read_csv("data/datatest.csv")
    X = data.drop(columns=['target','User_ID','Created At Year','Created At time',"Created At Month","Countries_ID"])
    Y = data['target']
    test_df = pd.read_csv("data/datatest.csv")
    test = test_df.select_dtypes(include=["number"]).drop(columns=["Created At Year","Created At Month","Countries_ID"])
    return X, Y, test

def train(X,Y,test,cv_splits=5):
    model1 = BalancedRandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=4,random_state=SEED)
    skfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    predictions1 = []

    for fold, (trn_idx, val_idx) in enumerate(skfold.split(X, Y)):
        print(f'BalancedRF Fold {fold + 1}')
        X_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_test  = X.iloc[val_idx], Y.iloc[val_idx]

        model1.fit(X_train, y_train)

        y_pred_valid = model1.predict(X_valid)
        oof[val_idx] = model1.predict_proba(X_valid)[:, 1]

        y_pred_train = model1.predict(X_train)

        # print("Training classification report:\n", classification_report(y_train, y_pred_train))
        # print("Validation classification report:\n", classification_report(y_test, y_pred_valid))

        predictions1.append(model1.predict_proba(test)[:, 1])

    predictions1 = np.mean(predictions1, axis=0)
    #X['rf'] = oof
    #test['rf'] = predictions1

    model1 = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(class_weight='balanced', max_iter=10000))
    ])

    skfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    predictions1 = []

    for fold, (trn_idx, val_idx) in enumerate(skfold.split(X, Y)):
        print(f'Logistic Regression Fold {fold + 1}')
        X_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_test  = X.iloc[val_idx], Y.iloc[val_idx]

        model1.fit(X_train, y_train)

        y_pred_valid = model1.predict(X_valid)
        oof[val_idx] = model1.predict_proba(X_valid)[:, 1]

        y_pred_train = model1.predict(X_train)

        # print("Training classification report:\n", classification_report(y_train, y_pred_train))
        # print("Validation classification report:\n", classification_report(y_test, y_pred_valid))

        predictions1.append(model1.predict_proba(test)[:, 1])

    predictions1 = np.mean(predictions1, axis=0)

    X['lg'] = oof
    test['lg'] = predictions1

    model1 = CatBoostClassifier(
        n_estimators=1000,
        learning_rate=0.0100800800100051124,
        depth=7,
        random_seed=0,
        auto_class_weights='Balanced',
        verbose=1
    )

    skfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    predictions1 = []


    for fold, (trn_idx, val_idx) in enumerate(skfold.split(X, Y)):
        print(f'CatBoost (stacking) Fold {fold + 1}')
        X_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_test  = X.iloc[val_idx], Y.iloc[val_idx]

        model1.fit(
            X_train, y_train,
            eval_set=(X_valid, y_test),
            verbose=100,
            early_stopping_rounds=100,
        )

        y_pred_valid = model1.predict(X_valid)

        oof[val_idx] = model1.predict_proba(X_valid)[:, 1]

        y_pred_train = model1.predict(X_train)

        # print("Training classification report:\n", classification_report(y_train, y_pred_train))
        # print("Validation classification report:\n", classification_report(y_test, y_pred_valid))

        predictions1.append(model1.predict_proba(test)[:, 1])

    predictions1 = np.mean(predictions1, axis=0)

    X['cat'] = oof
    test['cat'] = predictions1

    X['weighted_avg'] = X[['lg', 'cat']].mean(axis=1)
    test['weighted_avg'] = test[['lg', 'cat']].mean(axis=1)

    X['y'] = Y
    X['diff'] = X['y'] - X['cat']
    X = X[X['diff'] > -0.85]
    Y = X['y']
    X = X.drop(['diff', 'y'], axis=1)

    model1 = CatBoostClassifier(
        n_estimators=2000,
        learning_rate=0.001005,
        depth=9,
        random_seed=0,
        auto_class_weights='Balanced',
        verbose=0
    )

    skfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    predictions1 = []
    precision_per_fold = []
    recall_per_fold = []
    f1_per_fold = []
    precision_train_per_fold = []
    recall_train_per_fold = []
    f1_train_per_fold = []

    for fold, (trn_idx, val_idx) in enumerate(skfold.split(X, Y)):
        print(f'Final CatBoost Fold {fold + 1}')
        X_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_valid  = X.iloc[val_idx], Y.iloc[val_idx]

        model1.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            verbose=100,
            early_stopping_rounds=100,
        )
        oof[val_idx] = model1.predict(X_valid)
        y_pred_valid = model1.predict(X_valid)
        
        # Training predictions
        y_pred_train= model1.predict(X_train)
        #print("Training classification report:\n", classification_report(y_train, y_pred_train))
        
        # Validation report
        #print("Validation classification report:\n", classification_report(y_valid, y_pred_valid))
        
        y_pred_train = model1.predict(X_train)
        precision, recall, f1, _ = precision_recall_fscore_support(y_valid, y_pred_valid, average=None)
        precisiontrain, recalltrain, f1train, _ = precision_recall_fscore_support(y_train, y_pred_train, average=None)
        precision_per_fold.append(precision)
        recall_per_fold.append(recall)
        f1_per_fold.append(f1)
        precision_train_per_fold.append(precisiontrain)
        recall_train_per_fold.append(recalltrain)
        f1_train_per_fold.append(f1train)
        predictions1.append(model1.predict_proba(test)[:, 1])

    # Convert lists to numpy arrays
    precision_per_fold = np.array(precision_per_fold)
    recall_per_fold = np.array(recall_per_fold)
    f1_per_fold = np.array(f1_per_fold)
    precision_train_per_fold = np.array(precision_train_per_fold)
    recall_train_per_fold = np.array(recall_train_per_fold)
    f1_train_per_fold = np.array(f1_train_per_fold)

    avg_precision_per_class = np.mean(precision_per_fold, axis=0)
    avg_recall_per_class = np.mean(recall_per_fold, axis=0)
    avg_f1_per_class = np.mean(f1_per_fold, axis=0)
    avg_precision_train_per_class = np.mean(precision_train_per_fold, axis=0)
    avg_recall_train_per_class = np.mean(recall_train_per_fold, axis=0)
    avg_f1_train_per_class = np.mean(f1_train_per_fold, axis=0)
    
    for i, (p, r, f) in enumerate(zip(avg_precision_per_class, avg_recall_per_class, avg_f1_per_class)):
        print(f"Class {i}: Avg Val Precision={p:.4f}, Avg Val Recall={r:.4f}, Avg Val F1={f:.4f}")

    for i, (p, r, f) in enumerate(zip(avg_precision_train_per_class, avg_recall_train_per_class, avg_f1_train_per_class)):
        print(f"Class {i}: Avg Train Precision={p:.4f}, Avg Train Recall={r:.4f}, Avg Train F1={f:.4f}")

    print(f'Our Out Of Fold f1 score is {classification_report(Y, oof)}')

    predictions1 = np.mean(predictions1, axis=0)
    return predictions1,model1





