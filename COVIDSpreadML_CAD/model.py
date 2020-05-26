
from datetime import datetime as dt
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import pandas as pd
import shap
import numpy as np
import csv
import matplotlib.pyplot as plt
import pdb
import warnings
import statsmodels.tsa as tsa

import dataload


class FeaturesData:
  """
  Class for loading the data (in the form of Pandas Data Frames) and building feature vectors
  """
  def __init__(self):
    self.confirmed_cases_df, \
    self.deaths_df, \
    self.positive_tests_df, \
    self.negative_tests_df, \
    self.hospitals_df, \
    self.census_df, \
    self.cuebiqMI, \
    self.googleMobility, \
    self.unacast_distance_dff, \
    self.unacast_visitation_dff, \
    self.unacast_total, \
    self.distance_traveled, \
    self.education_df, \
    self.race_df, \
    self.employment_df = dataload.get_all_datasets()
  
  def get_feature_vector(self, county, current_date, lookback_date):
    features = []

    ### DT EDIT ###
    features += self.census_df.loc[county, 'Population':].to_list()
    featureNames = list(self.census_df.columns)
    #features += self.hospitals_df.loc[county, lookback_date:current_date].to_list()
    features += self.education_df.loc[county, :].to_list()  # use all features (not time-based)
    featureNames += list(self.education_df.columns)
    features += self.employment_df.loc[county, :].to_list()  # use all features (not time-based)
    featureNames += list(self.employment_df.columns)
    features += self.race_df.loc[county, :].to_list()  # use all features (not time-based)
    featureNames += list(self.race_df.columns)
    features += self.unacast_distance_dff.loc[county, lookback_date:current_date].to_list()
    features += self.unacast_visitation_dff.loc[county, lookback_date:current_date].to_list()
    features += self.deaths_df.loc[county, lookback_date:current_date].to_list()
    features += self.positive_tests_df.loc[county, lookback_date:current_date].to_list()
    features += self.negative_tests_df.loc[county, lookback_date:current_date].to_list()
    dates = list(self.deaths_df.loc[county, lookback_date:current_date].axes[0])

    features += self.cuebiqMI.loc[county, lookback_date:current_date].to_list()

    features += self.unacast_total.loc[county, lookback_date:current_date].to_list()
    #features += self.distance_traveled.loc[county, lookback_date:current_date].to_list()
    features += self.confirmed_cases_df.loc[county, lookback_date:current_date].to_list()

    featLabel = ['unacast_distance ','unacast_visitation ','deaths_df ', 'positive_tests_df ', 'negative_tests_df ','cuebiq_MI ' ,'unacast_total ', 'confirmed_cases_df ']
    for i in range(len(featLabel)):
      for date in dates:
        featureNames.append(featLabel[i] + str(date))
    return features, np.array(featureNames)

def build_training_dataset(
        days_of_history,
        future_prediction_days,
        obj_fun_type,
        start_prediction_date,
        last_prediction_date):
  """
  Builds the feature vectors and labels for training. See the 'run' function for description and reasonable default values
  :param days_of_history:
  :param future_prediction_days:
  :param obj_fun_type:
  :param start_prediction_date:
  :param last_prediction_date:
  :return:
  """

  assert(obj_fun_type in ['log', 'linear', 'log_relative_change'])
  assert(start_prediction_date < last_prediction_date)
  
  
  import warnings
  warnings.simplefilter('error')

  fd = FeaturesData()
  print('data for features loaded')
  
  feature_vectors = []
  labels = []

  feature_vectors_last_date = {}
  labels_last_date = {}

  feature_vectors_future = {}

  for county, county_data in fd.confirmed_cases_df.loc[:, start_prediction_date:last_prediction_date + timedelta(days=1)].iterrows():
    for prediction_date, val in county_data.iteritems():
      current_date = prediction_date - timedelta(days=future_prediction_days)
      lookback_date = current_date - timedelta(days=days_of_history)
      if fd.confirmed_cases_df.loc[county, current_date] >= 7:
        ### DT EDIT ###
        features, featureNames = fd.get_feature_vector(county, current_date, lookback_date)
        if prediction_date == last_prediction_date:
          feature_vectors_last_date[(county, prediction_date)] = features
          labels_last_date[(county, prediction_date)] = fd.confirmed_cases_df.loc[county, prediction_date]
        else:
          feature_vectors.append(features)
          if obj_fun_type == 'log':
            labels.append(np.log(1.0 + fd.confirmed_cases_df.loc[county, prediction_date]))
          elif obj_fun_type =='linear':
            labels.append(fd.confirmed_cases_df.loc[county, prediction_date])
          else:
            rel_increase = np.log(fd.confirmed_cases_df.loc[county, prediction_date]) - np.log(features[-1])
            labels.append(rel_increase)

    if fd.confirmed_cases_df.loc[county, last_prediction_date] >= 7:
      lookback_date = last_prediction_date - timedelta(days=days_of_history)
      try:
        features = fd.get_feature_vector(county, last_prediction_date, lookback_date)
        feature_vectors_future[(county, last_prediction_date + timedelta(days=future_prediction_days))] = \
          features
        
      except:
        print('ERROR in building features for forward prediction')
        pass
  
  print('Number of training examples=', len(feature_vectors))
  print('Number of testing examples on most recent date=', len(feature_vectors_last_date))
  print('Number of forward-looking predictions to be made=', len(feature_vectors_last_date))
  ### DT EDIT ###
  return feature_vectors, featureNames, labels, feature_vectors_last_date, labels_last_date, feature_vectors_future

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


from sklearn.base import clone 

def imp_df(column_names, importances):
  df = pd.DataFrame({'feature': column_names,'feature_importance': importances}).sort_values('feature_importance', ascending = False).reset_index(drop = True)
  return df

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
  # clone the model to have the exact same specification as the one initially trained
  model_clone = clone(model)
  # set random_state for comparability
  model_clone.random_state = random_state
  # training and scoring the benchmark model
  model_clone.fit(X_train, y_train)
  benchmark_score = model_clone.score(X_train, y_train)
  # list for storing feature importances
  importances = []
    
  # iterating over all columns and storing feature importance (difference between benchmark and new model)
  counter = 0
  for col in X_train.columns:
    counter += 1
    print('Feature Dropout: ' + str(counter))
    model_clone = clone(model)
    model_clone.random_state = random_state
    model_clone.fit(X_train.drop(col, axis = 1), y_train)
    drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
    importances.append(benchmark_score - drop_col_score)
    
  importances_df = imp_df(X_train.columns, importances)
  return importances_df

def train_and_test_model(feature_vectors, featureNames, labels,
                         feature_vectors_last_date, labels_last_date,
                         feature_vectors_future,
                         obj_fun_type,
                         predictions_output_filename,
                         classifier_fct
                         ):
  """
  Trains the model and outputs predictions. See the 'run' function for description and reasonable default values
  :param feature_vectors:
  :param labels:
  :param feature_vectors_last_date:
  :param labels_last_date:
  :param feature_vectors_future:
  :param obj_fun_type:
  :param predictions_output_filename:
  :paranm classifier_fct:
  :return:
  """
  # classifier = RandomForestRegressor(n_estimators=500)
  

  ### DT EDIT ###
  classifier = classifier_fct(n_estimators=500, oob_score = True)



  num_cross_folds = 5
  
  avg_abs_error = 0.0
  avg_relative_abs_error = 0.0
  count = 0.0

  for c in range(num_cross_folds):
    print('Running crossval, fold', c + 1)
    
    # Split the examples between training and testing
    # X_train = []
    # Y_train = []
    # X_test = []
    # Y_test = []
    # for i in range(len(feature_vectors)):
    #   if i % num_cross_folds == c:
    #     X_test.append(feature_vectors[i])
    #     Y_test.append(labels[i])
    #   else:
    #     X_train.append(feature_vectors[i])
    #     Y_train.append(labels[i])
    feature_vectors = np.array(feature_vectors)
    feature_df = pd.DataFrame(data = feature_vectors, columns = featureNames)

    X_train, X_test, Y_train, Y_test = train_test_split(feature_df, pd.Series(labels), test_size = 1/num_cross_folds)
    
    classifier.fit(X_train, Y_train)
    print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f}'.format(classifier.score(X_train, Y_train), classifier.oob_score_))  

    shap.initjs()

    sampleX = shap.sample(X_test, 100)
    sampleX = X_test
    # exp = shap.TreeExplainer(classifier, sampleX)
    # shap_values = exp.shap_values(X_train)
    # pdb.set_trace()
    # shap_values = shap.TreeExplainer(classifier, model_output = "raw").shap_interaction_values(sampleX)
    # shap_values = shap.KernelExplainer(classifier, sampleX).shap_values(sampleX)

    # # # pdb.set_trace()
    # # # f1 = plt.figure()
    # # # shap.summary_plot(shap_values, X_train, plot_type="bar")
    # # # f1.savefig('/shap_summary_value.png', bbox_inches = 'tight', dpi=600)
    # # # plt.show()

    # f2 = plt.figure()
    # shap.summary_plot(shap_values, sampleX)
    # f2.savefig('shap_variable_importance.png', bbox_inches = 'tight', dpi=600)
    # plt.show()
    # pdb.set_trace()
    featureImportance = None
    # featureImportance = drop_col_feat_imp(classifier, X_train, Y_train)

    # plt.figure()
    # featureImportance.plot.bar(x = 'feature', y = 'feature_importance')
    # plt.show()
    # pdb.set_trace()

    predictions = classifier.predict(X_test).tolist()

    for i, p in enumerate(predictions):
      if obj_fun_type == 'log':
        y_t = np.exp(Y_test[i]) - 1
        #to prevent overflow (required when using linear regression instead of Random Forests
        if p > 100:
          p = 100
        y_p = np.exp(p) - 1
      elif obj_fun_type == 'linear':
        y_t = Y_test[i]
        y_p = p
      else:
        # pdb.set_trace()
        y_t = np.exp(Y_test.tolist()[i]) * X_test[i][-1]
        y_p = np.exp(p) * X_test[i][-1]

      avg_abs_error += np.fabs(y_p - y_t)
      avg_relative_abs_error += np.fabs(y_p - y_t) / (y_t or 1.0)
      count += 1

      break

    # return featureImportance
  
  # featureImportance = permutation_importance(classifier, X_test, Y_test, n_repeats=30, random_state=0)

  output_row = {}
  county_info = dataload.get_county_info('data/covid_confirmed_cases.csv') #county data included in the output file

  for (county, date), xi in feature_vectors_last_date.items():
    predict = classifier.predict([xi])
    if obj_fun_type == 'log':
      if predict[0] > 100:
        predict[0] = 100
      p = np.exp(predict[0]) - 1
    elif obj_fun_type == 'linear':
      p = predict[0]
    else:
      # NOTE: it is assumed that the last feature is the number of cases for the most recent date in the feature set
      p = xi[-1] * np.exp(predict[0])

    if county not in output_row:
      output_row[county] = [None, None, None]
    output_row[county][0] = labels_last_date[(county, date)]
    output_row[county][1] = p

  for (county, date), xi in feature_vectors_future.items():
    ### DT EDIT ###
    xi = xi[0] ### Had to modify xi to xi[0] because it was being returned as a tuple with the new CUEBIQ DATA - potential bug?

    predict = classifier.predict([xi])
    if obj_fun_type == 'log':
      if predict[0] > 100:
        predict[0] = 100
      p = np.exp(predict[0]) - 1
    elif obj_fun_type == 'linear':
      p = predict[0]
    else:
      # NOTE: it is assumed that the last feature is the number of cases for the most recent date in the feature set
      p = xi[-1] * np.exp(predict[0])

    if county not in output_row:
      output_row[county] = [None, None, None]
    output_row[county][2] = p

  with open(predictions_output_filename, 'w') as fout_raw:
    fout = csv.writer(fout_raw)
    fout.writerow(
      ['countyFIP',
       'County Name',
       'State',
       'Most Recent Recorded Cases',
       'Predicted Most Recent Cases',
       'Future Prediction'])
    for county_id, v in output_row.items():
      row = [county_id, county_info[repr(county_id)]['County Name'], county_info[repr(county_id)]['State']] + v
      fout.writerow(row)
  
  print('Average absolute error = ', avg_abs_error / count)
  print('Average relative absolute = ', avg_relative_abs_error / count)

  # return featureImportance


def run():

  obj_fun_type='log_relative_change' #the type of objective function the model will predict
  """
  linear - the ML algorithm will try to predict the actual number of cases
  log - the algorithm will try to predict the log of the case count, and then convert back to linear to measure error
  log_relative_change - the algorithm will try to predict the log of the relative change in case count compared to the
   current case count (recommended)
  """
  predictions_output_filename = 'predictions.csv' #the name of the output file for the current and future predctions.

  days_of_history = 14 #how many days back the model will use to build features
  future_prediction_days = 14 #how far in advance the model is making predictions
  start_prediction_date = dt(2020, 3, 4).date() #the earliest date the model will use as a training example label
  last_prediction_date = dt(2020, 4, 7).date() #the last date the model will use as a training example label
  # last_prediction_date = dt(2020, 3, 25).date() #the last date the model will use as a training example label
  
  #the model will use all of the labels for the last_prediction_date as test cases where it will output the predictions
  # made for that date previously (future_prediction_days days prior)
  
  #the model will also make predictions for the date last_prediction_date + future_prediction_days and output those
  # predictions into the spreadsheet as well

  #build the training data
  feature_vectors, featureNames, labels, feature_vectors_last_date, labels_last_date, feature_vectors_future = \
    build_training_dataset(
      days_of_history,
      future_prediction_days,
      obj_fun_type,
      start_prediction_date,
      last_prediction_date
    )


  

  classifier_fct = RandomForestRegressor
  #train the model and output the predictions
  featureImportance = train_and_test_model(feature_vectors, featureNames,
                       labels,
                       feature_vectors_last_date,
                       labels_last_date,
                       feature_vectors_future,
                       obj_fun_type,
                       predictions_output_filename,
                       classifier_fct)


  # permutation importance
  # performance = featureImportance.importances_mean
  # sortedInd = np.argsort(performance)

  # featureNames = featureNames[sortedInd]
  # performance = performance[sortedInd]

  # y_pos = np.arange(len(featureNames[-20:]))

  # plt.barh(y_pos, performance[-20:], align='center', alpha=0.5)
  # plt.yticks(y_pos, featureNames[-20:])
  # plt.xlabel('Permutation Importance')
  # # plt.title('Programming language usage')

  # plt.show()


if __name__ == '__main__':
  run()
