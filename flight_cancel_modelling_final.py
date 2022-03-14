# ! pip install scikit-plot
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, log_loss
import time
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

start = time.process_time()

# Code is used for predicting cancelations
# Model tunning method is commented and can only be commented out when tunning is required it takes hours to complete

# The tunning is obtained and then the models are run in a loop with tunning parameters
# All the graphs in the reported are generated with these codes

# Note The code with delay predictions stays the same with different tunning and different target variable
# Hence this code can be considered as the final draft


# Calculates the Feature Importance for each model and plots a graph for it
def displayFeautureImportances(modelName, model, x_train_scaled, y_train, x_test_scaled,y_test):
    print('==================Top 30 Features=======================')
    feature_importance = {}
    if ("Logistic" in modelName):
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = model.coef_[0][i]
    elif ("Bagging" in modelName):
        feature_importances = np.mean([
            model.feature_importances_ for model in model.estimators_
        ], axis=0)
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = feature_importances[i]
    elif ("Neural" in modelName or "XGB" in modelName):
        print("Figure out")
        return
    else:
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = model.feature_importances_[i]

    feature_importance = dict(sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True))

    print(feature_importance)

    first30pairs = {k: feature_importance[k] for k in list(feature_importance)[:30]}
    print("Displaying top 30 Feature Importances")
    print(first30pairs)
    my_df = pd.DataFrame(first30pairs.items())
    plt.figure(figsize=(20.5, 10.5))
    ax = sns.barplot(x=1, y=0, color="dodgerblue", label="Features", data=my_df)
    ax.legend(loc='upper right', prop={'size': 9})
    ax.set_ylabel('Top Features', fontweight='bold', fontsize=10, labelpad=20)
    ax.set_xlabel('Importance Scores', fontweight='bold', fontsize=10, labelpad=20)
    plt.xticks(rotation=90)
    plt.title("Top 30 Important Feature", pad=20)
    plt.show()

# This is the final comparison graph which compares different measures such as Accuracy, AUC, Recall, Specificity etc
def plotPerformanceSummary(acc):
    accuracy = [metric.get("accuracy") for metric in acc]
    modelName = [metric.get("model") for metric in acc]
    roc_auc_score = [metric.get("roc_auc_score") for metric in acc]
    recall_score = [metric.get("recall_score") for metric in acc]
    precision_score = [metric.get("precision_score") for metric in acc]
    f1_score = [metric.get("f1_score") for metric in acc]
    specificity = [metric.get("specificity") for metric in acc]

    plt.figure(figsize=(30.5, 10.5))
    ax = sns.lineplot(x=modelName, y=accuracy, color="red", linestyle='-', marker="x", label="Accuracy")
    ax = sns.barplot(x=modelName, y=roc_auc_score, color="lightblue", label="AUC")
    ax = sns.lineplot(x=modelName, y=recall_score, color="royalblue", linestyle='--', marker="o",
                      label="Recall/Sensitivity")
    ax = sns.lineplot(x=modelName, y=f1_score, color="lightseagreen", linestyle='--', marker="o", label="F1_Score")
    ax = sns.lineplot(x=modelName, y=precision_score, color="gray", linestyle='--', marker="o", label="Precission")
    ax = sns.lineplot(x=modelName, y=specificity, color="black", linestyle='--', marker="o", label="Specificity")

    for lable, value in zip(ax.patches, roc_auc_score):
        _x = lable.get_x() + lable.get_width() / 2
        _y = lable.get_y() + lable.get_height() + 0.006
        ax.text(_x, _y, round(value, 3), ha="center", fontsize=12)

    def calculate_bar(ax, val):
        for bar in ax.patches:
            width = bar.get_width()
            diff = width - val
            bar.set_width(val)
            bar.set_x(bar.get_x() + diff * .5)

    calculate_bar(ax, .65)

    ax.legend(loc='upper left', prop={'size': 7.5})
    ax.set_ylabel('Relative Score Measures', fontweight='bold', fontsize=9.5, labelpad=19.5)
    ax.set_xlabel('Predicitve Models', fontweight='bold', fontsize=9.5, labelpad=19.5)
    plt.title("Performance of Various Predicitve Models", pad=20)
    plt.show()

# This is the method used for tunning with cross validartion, it's commented usually
def ModelTuning(tuning_params, x_train_scaled, x_test_scaled, y_train, y_test):
    for tuning in tuning_params:
        estimator = tuning.get("estimator")
        parameters = tuning.get("parameters")
        name = tuning.get("name")
        print('==================Performing Tuning for ' + name + '=======================')
        strat_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
        grid_tuned = GridSearchCV(estimator=estimator, param_grid=parameters, cv=strat_fold, scoring='roc_auc',
                                   refit=True)
        grid_tuned.fit(x_train_scaled, y_train.values.ravel())
        prediction = grid_tuned.predict(x_test_scaled)
        prediction_prob = grid_tuned.predict_proba(x_test_scaled)
        best_classifier = grid_tuned.best_estimator_
        print('estimator:', estimator, 'Best Classifier:', best_classifier)
        print(' Accuracy Score:', accuracy_score(y_test, prediction))
        print(' ROC AUC:', roc_auc_score(y_test, prediction_prob[:, 1]))
        means = grid_tuned.cv_results_['mean_test_score']
        stds = grid_tuned.cv_results_['std_test_score']

        print('Mean AUC (+/- standard deviation), for parameters')
        for mean, std, params in zip(means, stds, grid_tuned.cv_results_['params']):
            print("%0.3f (+/- %0.03f) for %r"
                  % (mean, std, params))

# Method for SVN tunning
def SVNTuning(x_train_set, x_test_set, y_train, y_test):
    parameters = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.2, 0.5, 1.0]}
    strat_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    grid_tuned = GridSearchCV(SVC(gamma='auto'), parameters, cv=strat_fold, scoring='roc_auc')
    grid_tuned.fit(x_train_set, y_train.values.ravel())
    mean_score = grid_tuned.cv_results_['mean_test_score']
    st_score = grid_tuned.cv_results_['std_test_score']
    print('Mean AUC (+/- standard deviation), for parameters')
    for mean, std, params in zip(mean_score, st_score, grid_tuned.cv_results_['params']):
        print("%0.3f (+/- %0.03f) for %r"
              % (mean, std, params))

# Method for running SVN alsonas its time consuming
def SVNRun(x_train_set, x_test_set, y_train, y_test):
    model = SVC(gamma='auto', C=0.1, kernel='rbf', probability=True)
    model.fit(x_train_set, y_train.values.ravel())
    prediction = model.predict(x_test_set)
    # pprint(prediction)
    truneg, falpos, falneg, trupos = confusion_matrix(y_test, prediction).ravel()
    specifi_score = truneg / (truneg+falpos)
    prediction_prob = model.predict_proba(x_test_set)
    print('==================Metrics for Test Set=======================')
    print('Accuracy:', accuracy_score(y_test, prediction))
    print('ROC AUC:', roc_auc_score(y_test, prediction_prob[:,1]))
    print('Recall:', recall_score(y_test, prediction))
    print('Precision:', precision_score(y_test, prediction))
    print('f1_score:', f1_score(y_test, prediction))
    print('f1_score:', f1_score(y_test, prediction))
    print('specificity:', specifi_score)
    print('==================ROC Curve for Test Set=======================')
    skplt.metrics.plot_roc_curve(y_test, prediction_prob[:,1])
    plt.title('ROC Curve for SVM')
    plt.show()
    feature_importance = {}
    for i in range(0, len(x_test_set.columns)):
        feature_importance[x_test_set.columns[i]] = model.coef_[0][i]
    feature_importance = dict(sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True))
    first30pairs = {k: feature_importance[k] for k in list(feature_importance)[:30]}
    print("Displaying top 30 Feature Importances")
    print(first30pairs)
    plt.figure(figsize=(20, 14))
    plt.bar(range(len(first30pairs)), first30pairs.values())
    plt.xticks(range(len(first30pairs)), list(first30pairs.keys()))
    plt.xticks(rotation=90)
    plt.show()
    print("Graphs displayed")

# Seperate method is used to perform lass regression with logistic
def Tune_Lasso_Logistic(x_train_scaled, y_train):
    lasso_logistic_model = LogisticRegression(
        penalty='l1',
        solver='liblinear')

    grid = dict()
    grid['C'] = arange(0.0001, 1, 0.01)

    grid_tune = GridSearchCV(lasso_logistic_model, grid, scoring='accuracy', cv=5, refit=True)
    results = grid_tune.fit(x_train_scaled, y_train)
    best_params = results.best_params_

    print('Config: %s' % results.best_params_)
    return best_params

# This method runs all the models with parameters passed predicts and plots the graphs
def ModelPredictionAndAccuracy(modelling_params, x_train_scaled, x_test_scaled, y_train, y_test):
    acc = []
    for modelparams in modelling_params:
        model = modelparams.get('model')
        modelName = modelparams.get('name')
        print('==================Predicting Model:', modelName, '=======================')
        prediction = model.predict(x_test_scaled)
        if ("Neural" in modelName):
            prediction_nn = (prediction > 0.5).astype('int32')

            def predict_prob_f(number):
                return [1 - number[0], number[0]]

            prediction_prob = np.array(list(map(predict_prob_f, model.predict(x_test_scaled))))
            prediction = prediction_nn
        else:
            prediction_prob = model.predict_proba(x_test_scaled)

        metrics_list = {}
        print('==================Metrics for Test Set=======================')
        modelNameGR = modelparams.get('grname')
        truneg, falpos, falneg, trupos = confusion_matrix(y_test, prediction).ravel()
        speci_score = truneg / (truneg+falpos)
        print('Accuracy:', accuracy_score(y_test, prediction))
        print('ROC AUC Score:', roc_auc_score(y_test, prediction_prob[:,1]))
        print('Recall/Sensitivity:', recall_score(y_test, prediction))
        print('Precision:', precision_score(y_test, prediction))
        print('f1_score:', f1_score(y_test, prediction))
        print('specificity:', speci_score)
        metrics_list['accuracy'] = accuracy_score(y_test, prediction)
        metrics_list['roc_auc_score'] = roc_auc_score(y_test, prediction_prob[:,1])
        metrics_list['recall_score'] = recall_score(y_test, prediction)
        metrics_list['precision_score'] = precision_score(y_test, prediction)
        metrics_list['f1_score'] = f1_score(y_test, prediction)
        metrics_list['specificity'] = speci_score
        metrics_list['model'] = modelNameGR
        acc.append(metrics_list)
        print('==================ROC Curve for Test Set=======================')
        skplt.metrics.plot_roc_curve(y_test, prediction_prob)
        plt.title('ROC Curve for ' + modelName)
        plt.show()

        displayFeautureImportances(modelName, model, x_train_scaled, y_train, x_test_scaled,y_test)

    return acc

# This method is used for normalization with scalling done on train test sets,
# all the tuned parameters are passed for running models
def ModelTuning_Predition(flights, scenario, col_names):
    print("Running Scenario prediciton :", scenario)
    y = flights['Cancelled']
    X = flights.drop('Cancelled', axis=1)

    # Data splitting with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.3, random_state=42)

    # Scale the training and the test data
    x_train_s = X_train.copy()
    x_test_scaled = X_test.copy()

    # Min Max Scalling
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_feat = x_train_s[col_names]
    train_feat_f = min_max_scaler.fit_transform(train_feat.values)
    x_train_s[col_names] = train_feat_f
    test_features = x_test_scaled[col_names]
    test_features_f = min_max_scaler.fit(train_feat.values).transform(test_features.values)
    x_test_scaled[col_names] = test_features_f

    print("Cancelled Train Data")
    print(y_train.value_counts())

    # Application of smote
    sm = SMOTE()
    x_train_scaled, y_train = sm.fit_resample(x_train_s, y_train)

    print("Cancelled Train Data post Smote")
    print(y_train.value_counts())
    print("Data is Cleaned")
    print("Starting the Classification Model Run...")

    input_dim = x_train_scaled.shape[1]
    output_dim = 1

# Method is used for tunning the Neural models
    def nn_model(no_neurons, learning_rate, no_layers, kernel):
        model = Sequential()
        model.add(Dense(no_neurons, input_dim=input_dim))
        model.add(Activation(kernel))

        # Extra hidden layers
        for _ in range(0, no_layers):
            model.add(Dense(no_neurons))
        model.add(Activation(kernel))

        # Output
        model.add(Dense(output_dim))
        model.add(Activation('sigmoid'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # best_param = Tune_Lasso_Logistic(x_train_scaled, y_train)

# Tunning parameter ranges are passed for various models
    tuning_params = [
        {"estimator": DecisionTreeClassifier(random_state=44),
         "parameters": {'min_samples_leaf': [1, 5], 'max_depth': [None, 10]}
            , "name": "DecisionTreeClassifier with Stratification"},
        {"estimator": RandomForestClassifier(random_state=44),
         "parameters": {'min_samples_leaf': [1, 5], 'max_depth': [None, 10], 'n_estimators': range(10, 100, 10)}
            , "name": "Random Forest Classification with Stratification"},
        {"estimator": AdaBoostClassifier(random_state=44), "parameters": {'n_estimators': range(10, 100, 10)}
            , "name": "AdaBoostClassifier with Stratification"},
        {"estimator": BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=44),
         "parameters": {'n_estimators': range(40, 70, 10)}
            , "name": "BaggingClassifier with Stratification"},
        {"estimator": KerasClassifier(nn_model),
         "parameters": {'no_neurons': [50, 100], 'kernel': ['relu', 'sigmoid'], 'no_layers': [1, 2],
                        'learning_rate': [0.1, 0.01, 0.001], 'epochs': [10], 'verbose': [0]}
            , "name": "BaggingClassifier with Stratification"}
    ]

    # As its a time consuming process its ran before hand
    # ModelTuning(tuning_params, x_train_scaled, x_test_scaled, y_train, y_test)
    print("Model Tuning Completed before the process")
    print("Proceeding for Fitting the Models ")

    # Model Fitting done prior with tunned parameters
    logi_model = LogisticRegression(penalty='none').fit(x_train_scaled, y_train)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.9901).fit(x_train_scaled, y_train)
    decision_model = DecisionTreeClassifier(random_state=44).fit(x_train_scaled, y_train)
    decision_model_tuned = DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, random_state=44).fit(x_train_scaled,
                                                                                                         y_train)
    random_model = RandomForestClassifier(random_state=44).fit(x_train_scaled, y_train)
    random_forest_model_tuned = RandomForestClassifier(min_samples_leaf=5, max_depth=10, n_estimators=90,
                                                       random_state=44).fit(
        x_train_scaled, y_train)
    ada_booster_model = AdaBoostClassifier(random_state=44).fit(x_train_scaled, y_train)
    ada_booster_tuned = AdaBoostClassifier(n_estimators=90, random_state=44).fit(x_train_scaled, y_train)
    bagging_model = BaggingClassifier(random_state=44).fit(x_train_scaled, y_train)
    bagging_model_tuned = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=70,
                                            random_state=44).fit(x_train_scaled, y_train)
    xgb_model = XGBClassifier(random_state=44).fit(x_train_scaled, y_train)


    print("Neural network Summary")
    input_dim = X_train.shape[1]
    output_dim = 1

    nn_linear = Sequential()
    nn_linear.add(Dense(50, input_dim=input_dim))
    nn_linear.add(Activation('linear'))
    nn_linear.add(Dense(output_dim))
    nn_linear.add(Activation('sigmoid'))
    nn_linear.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nn_linear.summary())
    nn_linear.fit(x_train_scaled, y_train, epochs=10)

    nm_relu = Sequential()
    nm_relu.add(Dense(50, input_dim=input_dim))
    nm_relu.add(Activation('relu'))
    nm_relu.add(Dense(output_dim))
    nm_relu.add(Activation('sigmoid'))
    nm_relu.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nm_relu.summary())
    nm_relu.fit(x_train_scaled, y_train, epochs=10)

    nm_sigmoid = Sequential()
    nm_sigmoid.add(Dense(50, input_dim=input_dim))
    nm_sigmoid.add(Activation('sigmoid'))
    nm_sigmoid.add(Dense(50))
    nm_sigmoid.add(Activation('sigmoid'))
    nm_sigmoid.add(Dense(output_dim))
    nm_sigmoid.add(Activation('sigmoid'))
    nm_sigmoid.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nm_sigmoid.summary())
    nm_sigmoid.fit(x_train_scaled, y_train, epochs=10)

    # Fitted Models are added in parameter list to execute in sequence
    modelling_params = [
        {"model": logi_model, "name": "Logistic Regression without Tunning", "grname": "Logistic"},
        {"model": lasso_model, "name": "Logistic Regression with Lasso Tunning", "grname": "LogisticLasso"},
        {"model": decision_model, "name": "Decision Tree Classification", "grname": "Decision"},
        {"model": decision_model_tuned, "name": "Decision Tree Classification with Tunning", "grname": "DeciTuned"},
        {"model": random_model, "name": "Random Forest Classification", "grname": "RandomForest"},
        {"model": random_forest_model_tuned, "name": "Random Forest Classification with Tunning","grname": "RandomTuned"},
        {"model": ada_booster_model, "name": "ADA Booster Classification", "grname": "ADABooster"},
        {"model": ada_booster_tuned, "name": "ADA Booster Classification with Tunning", "grname": "ADABoosterTuned"},
        {"model": bagging_model, "name": "Bagging Model Classification", "grname": "Bagging"},
        {"model": bagging_model_tuned, "name": "Bagging Model with Tunning", "grname": "BaggingTuned"},
        {"model": xgb_model, "name": "XGBoost Model", "grname": "XGB"},
        {"model": nn_linear, "name": "Neural Model with Linear Activation", "grname": "NMLinear"},
        {"model": nm_sigmoid, "name": "Neural Model with Sigmoid Activation", "grname": "NMSigmoid"},
        {"model": nm_relu, "name": "Neural Model with Relu Activation", "grname": "NMRelu"}
    ]
    print("Models Fitted Successfully")

    # Predicts all the models and generates the metrics summary
    acc = ModelPredictionAndAccuracy(modelling_params, x_train_scaled, x_test_scaled, y_train, y_test)

    # Plots the finally summary graph with respect to AUC, Accuracy, Recall, Specificity
    plotPerformanceSummary(acc)

# Main flow to start the class
print("Starting taken", time.time())

# Fetch the the cleaned Data from the transformed data stored
flights = pd.read_csv('pre_processed_cancel_airline_na_dropped.csv')
flights.reset_index(drop=True, inplace=True)
# Drops unknwon coloumn if any
flights = flights.loc[:, ~flights.columns.str.contains('^Unnamed')]

# Drops the Cancelation_N Node as it is used in case of non cancelations
flights = flights.drop(['CancellationCode_N'], axis=1)
print(flights.keys())

# flights = flights.drop(['DepDelay'], axis=1)
col_names = ['Distance']
print(flights.keys())

# The method is called to loop over all the models with parameter tunning
ModelTuning_Predition(flights, "Model does not include Departure Delays", col_names)

print("Model Run Completed")
print("Time taken", time.process_time() - start)
