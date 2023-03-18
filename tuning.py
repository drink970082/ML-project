import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def tuning():

    # Logistic
    tree_max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    tree_max_depth.append(None)
    grid={"C":np.logspace(-5,5,10)}# l1 lasso l2 ridge
    logreg_cv=RandomizedSearchCV(estimator=LogisticRegression(),
                                param_distributions=grid, n_iter=20,verbose=0, n_jobs=-1)
    #bagging
    param_distributions = {
        'base_estimator__max_depth' : tree_max_depth,
        'max_samples' : [0.05, 0.1, 0.2, 0.5]
    }
    bag = RandomizedSearchCV(estimator=BaggingClassifier(DecisionTreeClassifier()),
                                            param_distributions=param_distributions, n_iter=20,verbose=0, n_jobs=-1)

    #Gradient Boosting
    parameters = {'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
                "n_estimators" : [130, 180, 230],
                "max_depth"    : tree_max_depth}
    grad = RandomizedSearchCV(estimator=GradientBoostingClassifier(), 
                            param_distributions = parameters, n_iter=20,verbose=0, n_jobs=-1)
    #rnd forest
    random_grid = {'bootstrap': [True, False],
                'max_depth': tree_max_depth,
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [130, 180, 230]}

    rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid,
                                n_iter=20,verbose=0, n_jobs=-1)
    # XGboost
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = np.arange(1,10)
    learning_rate=[round(float(x),2) for x in np.linspace(start=0.01, stop=0.2, num=10)]
    colsample_bytree =[round(float(x),2) for x in np.linspace(start=0.1, stop=1, num=10)]
    random_grid = {'n_estimators': n_estimators,
                'max_depth': tree_max_depth,
                'learning_rate': learning_rate,
                'colsample_bytree': colsample_bytree}
    xg_random = RandomizedSearchCV(estimator = XGBClassifier(), param_distributions=random_grid,
                                n_iter=20,verbose=0, n_jobs=-1)
    #adaboost
    parameters = {
        'n_estimators': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
        'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__max_depth' : tree_max_depth,
    }
    ada = RandomizedSearchCV(AdaBoostClassifier(DecisionTreeClassifier()), param_distributions=parameters
                            , n_iter=20,verbose=0, n_jobs=-1)
    #svc

    grid_list = {"C": np.arange(2, 10, 2),
                "gamma": np.arange(0.1, 1, 0.2)}
    svc = RandomizedSearchCV(svm.SVC(), param_distributions=grid_list, n_iter=20,verbose=0, n_jobs=-1)

    grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
                'weights' : ['uniform','distance'],
                'metric' : ['minkowski','euclidean','manhattan']}
    gs = RandomizedSearchCV(KNeighborsClassifier(), grid_params, n_iter=20,verbose = 0, n_jobs = -1)


    method_tuning = [logreg_cv,grad, bag,rf_random,
            ada, xg_random,svc, gs]
    method = [LogisticRegression(),GradientBoostingClassifier(), BaggingClassifier(),RandomForestClassifier(),
            AdaBoostClassifier(), XGBClassifier(),svm.SVC(), KNeighborsClassifier(n_neighbors=14)]
            
    method_name = ["Logistic Regression","Gradient Boosting Classifier", "Bagging Classifier"
                ,"Random Forest Classifier", "AdaBoost Classifier", "XGBoost Classifier",
                "Support Vector Classifier", "K Neighbors Classifier"]
    return method_name,method,method_tuning