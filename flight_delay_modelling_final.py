# ! pip install scikit-plot
import operator
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from numpy import arange
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import accuracy_score as accuracy
import time

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=sys.maxsize)

start = time.process_time()

def displayFeautureImportances(modelName, model, x_train_scaled, y_train):
    print('==================Top 30 Features=======================')
    feature_importance = {}
    if ("Logistic" in modelName):
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = model.coef_[0][i]
    elif ("Bagging" in modelName):
        feature_importances = np.mean([
            tree.feature_importances_ for tree in model.estimators_
        ], axis=0)
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = feature_importances[i]
    # elif("XGB" in modelName):
    #     plot_importance(model)
    elif ("Neural" in modelName or "XGB" in modelName):
        # perm = PermutationImportance(model, random_state=1).fit(x_train_scaled, y_train)
        # eli5.show_weights(perm, feature_names=y_train.columns.tolist())

        # explainer = shap.Explainer(model)
        # shap_values = explainer(x_train_scaled)
        # # visualize the first prediction's explanation
        # shap.plots.waterfall(shap_values[0])
        print("Figure out")
        return
    else:
        for i in range(0, len(model.feature_names_in_)):
            feature_importance[model.feature_names_in_[i]] = model.feature_importances_[i]

    feature_importance = dict(sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True))
    first30pairs = {k: feature_importance[k] for k in list(feature_importance)[:30]}
    print("Displaying top 30 Feature Importances")
    print(first30pairs)
    my_df = pd.DataFrame(first30pairs.items())
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=1, y=0, color="dodgerblue", label="Features", data=my_df)
    ax.legend(loc='upper right', prop={'size': 9})
    ax.set_ylabel('Top Features', fontweight='bold', fontsize=10, labelpad=20)
    ax.set_xlabel('Importance Scores', fontweight='bold', fontsize=10, labelpad=20)
    plt.xticks(rotation=90)
    plt.title("Top 30 Important Feature", pad=20)
    plt.show()


def plotPerformanceSummary(acc):
    accuracy = [metric.get("accuracy") for metric in acc]
    modelName = [metric.get("model") for metric in acc]
    roc_auc_score = [metric.get("roc_auc_score") for metric in acc]
    recall_score = [metric.get("recall_score") for metric in acc]
    precision_score = [metric.get("precision_score") for metric in acc]
    f1_score = [metric.get("f1_score") for metric in acc]
    specificity = [metric.get("specificity") for metric in acc]

    print(accuracy)
    print(modelName)
    print(roc_auc_score)
    print(recall_score)
    print(precision_score)
    print(f1_score)
    print(specificity)
    print(acc)

    plt.figure(figsize=(30, 10))
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

    ax.legend(loc='upper left', prop={'size': 8})
    ax.set_ylabel('Relative Score Measures', fontweight='bold', fontsize=10, labelpad=20)
    ax.set_xlabel('Predicitve Models', fontweight='bold', fontsize=10, labelpad=20)
    plt.title("Performance of Various Predicitve Models", pad=20)
    plt.show()

def ModelTuning(tuning_params, x_train_scaled, x_test_scaled, y_train, y_test):
    for tuning in tuning_params:
        estimator = tuning.get("estimator")
        parameters = tuning.get("parameters")
        name = tuning.get("name")
        print('==================Performing Tuning for ' + name + '=======================')
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
        grid_search = GridSearchCV(estimator=estimator, parameters=parameters, cv=stratified_kfold, scoring='roc_auc',
                                   refit=True)
        grid_search.fit(x_train_scaled, y_train.values.ravel())
        prediction = grid_search.predict(x_test_scaled)
        prediction_prob = grid_search.predict_proba(x_test_scaled)
        best_classifier = grid_search.best_estimator_
        print('estimator:', estimator, 'Best Classifier:', best_classifier)
        print(' Accuracy:', accuracy(y_test, prediction))
        print(' AUC:', roc_auc_score(y_test, prediction_prob[:, 1]))

def SVNTuning(x_train_scaled, x_test_scaled, y_train, y_test):
    parameters = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.2, 0.5, 1.0]}
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    grid_search = GridSearchCV(SVC(gamma='auto'), parameters, cv=stratified_kfold, scoring='roc_auc')
    grid_search.fit(x_train_scaled, y_train.values.ravel())
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    print('Mean AUC (+/- standard deviation), for parameters')
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/- %0.03f) for %r"
              % (mean, std, params))

def SVNRun(x_train_scaled, x_test_scaled, y_train, y_test):
    model = SVC(gamma='auto', C=0.1, kernel='rbf', probability=True)
    model.fit(x_train_scaled, y_train.values.ravel())
    prediction = model.predict(x_test_scaled)
    # pprint(prediction)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    specificity = tn / (tn+fp)
    prediction_prob = model.predict_proba(x_test_scaled)
    print('==================Metrics for Test Set=======================')
    print('Accuracy:', accuracy(y_test, prediction))
    print('ROC AUC:', roc_auc_score(y_test, prediction_prob))
    print('Recall:', recall_score(y_test, prediction))
    print('Precision:', precision_score(y_test, prediction))
    print('f1_score:', f1_score(y_test, prediction))
    print('f1_score:', f1_score(y_test, prediction))
    print('specificity:', specificity)
    print('==================ROC Curve for Test Set=======================')
    skplt.metrics.plot_roc_curve(y_test, prediction_prob)
    plt.title('ROC Curve for SVM')
    plt.show()
    plot_confusion_matrix(model, x_test_scaled, y_test)
    plt.show()
    feature_importance = {}
    for i in range(0, len(x_test_scaled.columns)):
        feature_importance[x_test_scaled.columns[i]] = model.coef_[0][i]
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

def Tune_Lasso_Logistic(x_train_scaled, y_train):
    lasso_logistic_model = LogisticRegression(
        penalty='l1',
        solver='liblinear')

    grid = dict()
    grid['C'] = arange(0.0001, 1, 0.01)

    search = GridSearchCV(lasso_logistic_model, grid, scoring='accuracy', cv=5, refit=True)
    results = search.fit(x_train_scaled, y_train)
    best_params = results.best_params_

    print('Config: %s' % results.best_params_)
    return best_params

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
        fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction)
        accu = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        specificity = tn / (tn+fp)
        print('Accuracy:', accuracy(y_test, prediction))
        print('AUC Score:', accu)
        print('ROC AUC Score:', roc_auc_score(y_test, prediction_prob[:,1]))
        print('Recall/Sensitivity:', recall_score(y_test, prediction))
        print('Precision:', precision_score(y_test, prediction))
        print('f1_score:', f1_score(y_test, prediction))
        print('specificity:', specificity)
        metrics_list['accuracy'] = accuracy(y_test, prediction)
        metrics_list['auc_score'] = accu
        metrics_list['roc_auc_score'] = roc_auc_score(y_test, prediction)
        metrics_list['recall_score'] = recall_score(y_test, prediction)
        metrics_list['precision_score'] = precision_score(y_test, prediction)
        metrics_list['f1_score'] = f1_score(y_test, prediction)
        metrics_list['specificity'] = specificity
        metrics_list['model'] = modelNameGR
        acc.append(metrics_list)
        print('==================ROC Curve for Test Set=======================')
        skplt.metrics.plot_roc_curve(y_test, prediction_prob)
        plt.title('ROC Curve for ' + modelName)
        plt.show()
        plot_confusion_matrix(model, x_test_scaled, y_test)
        plt.show()
        # losses = [log_loss(y_test, [y for x in range(len(prediction))]) for y in prediction]
        # plt.plot(prediction, losses)
        # plt.show()
        displayFeautureImportances(modelName, model, x_train_scaled, y_train)
    return acc

def ModelTuning_Predition(flights, scenario, col_names):
    print("Running Scenario prediciton :", scenario)
    y = flights['FlightDelayStatus']
    X = flights.drop('FlightDelayStatus', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.3, random_state=42)

    # Scale the training and the test data
    x_train_scaled = X_train.copy()
    x_test_scaled = X_test.copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_features = x_train_scaled[col_names]
    train_features_f = scaler.fit_transform(train_features.values)
    x_train_scaled[col_names] = train_features_f
    test_features = x_test_scaled[col_names]
    test_features_f = scaler.fit(train_features.values).transform(test_features.values)
    x_test_scaled[col_names] = test_features_f
    print("Data is Cleaned")
    print("Starting the Classification Model Run...")
    # SVNRun(x_train_scaled, x_test_scaled, y_train, y_test)

    input_dim = x_train_scaled.shape[1]
    output_dim = 1

    def nn_model_f(no_neurons, learning_rate, no_layers, kernel):
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

    tuning_params = [
        {"estimator": LogisticRegression(penalty='l1', solver='liblinear'),
         "parameters": {'n_estimators': range(10, 100, 10)}
            , "name": "Logistic Regression with Lasso"},
        {"estimator": DecisionTreeClassifier(random_state=44),
         "parameters": {'min_samples_leaf': [1, 5], 'max_depth': [None, 10]}
            , "name": "DecisionTreeClassifier with Stratification"},
        {"estimator": RandomForestClassifier(random_state=44),
         "parameters": {'min_samples_leaf': [1, 5], 'max_depth': [None, 10], 'n_estimators': range(10, 100, 10)}
            , "name": "Random Forest Classification with Stratification"},
        {"estimator": AdaBoostClassifier(random_state=44), "parameters": {'n_estimators': range(10, 100, 10)}
            , "name": "AdaBoostClassifier with Stratification"},
        {"estimator": BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=44),
         "parameters": {'n_estimators': range(10, 100, 10)}
            , "name": "BaggingClassifier with Stratification"}
    ]

    # As its a time consuming process its ran before hand
    # ModelTuning(tuning_params, x_train_scaled, x_test_scaled, y_train, y_test)
    print("Model Tuning Completed before the process")
    print("Proceeding for Fitting the Models ")

    logi_model = LogisticRegression(penalty='none').fit(x_train_scaled, y_train)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.9601).fit(x_train_scaled, y_train)
    decision_model = DecisionTreeClassifier(random_state=44).fit(x_train_scaled, y_train)
    decision_model_tuned = DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, random_state=44).fit(x_train_scaled,
                                                                                                         y_train)
    random_model = RandomForestClassifier(random_state=44).fit(x_train_scaled, y_train)
    random_forest_model_tuned = RandomForestClassifier(min_samples_leaf=5, n_estimators=40, random_state=44).fit(
        x_train_scaled, y_train)
    ada_booster_model = AdaBoostClassifier(random_state=44).fit(x_train_scaled, y_train)
    ada_booster_tuned = AdaBoostClassifier(n_estimators=90, random_state=44).fit(x_train_scaled, y_train)
    bagging_model = BaggingClassifier(random_state=44).fit(x_train_scaled, y_train)
    bagging_model_tuned = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=90,
                                            random_state=44).fit(x_train_scaled, y_train)
    xgb_model = XGBClassifier(random_state=44).fit(x_train_scaled, y_train)

    print("Neural network Summary")
    input_dim = X_train.shape[1]
    output_dim = 1

    nn_model_linear = Sequential()
    nn_model_linear.add(Dense(50, input_dim=input_dim))
    nn_model_linear.add(Activation('linear'))
    nn_model_linear.add(Dense(output_dim))
    nn_model_linear.add(Activation('sigmoid'))
    nn_model_linear.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nn_model_linear.summary())
    nn_model_linear.fit(x_train_scaled, y_train, epochs=10)

    nn_model_relu = Sequential()
    nn_model_relu.add(Dense(50, input_dim=input_dim))
    nn_model_relu.add(Activation('relu'))
    nn_model_relu.add(Dense(output_dim))
    nn_model_relu.add(Activation('sigmoid'))
    nn_model_relu.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nn_model_relu.summary())
    nn_model_relu.fit(x_train_scaled, y_train, epochs=10)

    nn_model_sigmoid = Sequential()
    nn_model_sigmoid.add(Dense(50, input_dim=input_dim))
    nn_model_sigmoid.add(Activation('sigmoid'))
    nn_model_sigmoid.add(Dense(50))
    nn_model_sigmoid.add(Activation('sigmoid'))
    nn_model_sigmoid.add(Dense(output_dim))
    nn_model_sigmoid.add(Activation('sigmoid'))
    nn_model_sigmoid.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(nn_model_sigmoid.summary())
    nn_model_sigmoid.fit(x_train_scaled, y_train, epochs=10)

    modelling_params = [
        {"model": logi_model, "name": "Logistic Regression without Tunning","grname": "Logistic"},
        {"model": lasso_model, "name": "Logistic Regression with Lasso Tunning","grname": "LogisticLasso"},
        {"model": decision_model, "name": "Decision Tree Classification","grname": "Decision"},
        {"model": decision_model_tuned, "name": "Decision Tree Classification with Tunning","grname": "DeciTuned"},
        {"model": random_model, "name": "Random Forest Classification","grname": "RandomForest"},
        {"model": random_forest_model_tuned, "name": "Random Forest Classification with Tunning","grname": "RandomTuned"},
        {"model": ada_booster_model, "name": "ADA Booster Classification","grname": "ADABooster"},
        {"model": ada_booster_tuned, "name": "ADA Booster Classification with Tunning","grname": "ADABoosterTuned"},
        {"model": bagging_model, "name": "Bagging Model Classification","grname": "Bagging"},
        {"model": bagging_model_tuned, "name": "Bagging Model with Tunning","grname": "BaggingTuned"},
        {"model": xgb_model, "name": "XGBoost Model", "grname": "XGB"},
        {"model": nn_model_linear, "name": "Neural Model with Linear Activation","grname": "NMLinear"},
        {"model": nn_model_sigmoid, "name": "Neural Model with Sigmoid Activation","grname": "NMSigmoid"},
        {"model": nn_model_relu, "name": "Neural Model with Relu Activation","grname": "NMRelu"}
    ]
    print("Models Fitted Successfully")

    acc = ModelPredictionAndAccuracy(modelling_params, x_train_scaled, x_test_scaled, y_train, y_test)
    plotPerformanceSummary(acc)

print("Starting taken", time.time())

# Fetch the the cleaned Data
flights = pd.read_csv('pre_processed_data_airline_na_dropped.csv')
flights.reset_index(drop=True, inplace=True)
flights = flights.loc[:, ~flights.columns.str.contains('^Unnamed')]
print(flights.keys())

col_names = ['DepDelay', 'Distance']
ModelTuning_Predition(flights, "Model include Departure Delays", col_names)
flights = flights.drop(['DepDelay'], axis=1)

col_names = ['Distance']
print(flights.keys())
ModelTuning_Predition(flights, "Model does not include Departure Delays", col_names)

print("Model Run Completed")
print("Time taken", time.process_time() - start)
