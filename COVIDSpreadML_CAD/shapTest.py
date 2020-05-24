import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
df = pd.read_csv('winequality-red.csv') # Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

import shap
import pdb

# The target variable is 'quality'.
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]

pdb.set_trace()

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Build the model with the random forest regression algorithm:
# model = RandomForestRegressor(n_estimators=500, oob_score = True)
# model.fit(X_train, Y_train)

# shap_values = shap.TreeExplainer(model).shap_values(X_train)
# # shap.summary_plot(shap_values, X_train, plot_type="bar")


# import matplotlib.pyplot as plt
# f = plt.figure()
# shap.summary_plot(shap_values, X_train)
# f.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)

    #X_train, X_test, Y_train, Y_test = train_test_split(feature_df, pd.Series(labels), test_size = 1/num_cross_folds)
    

classifier = RandomForestRegressor(n_estimators=500, oob_score = True)
classifier.fit(X_train, Y_train)
    # print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f}'.format(classifier.score(X_train, Y_train), classifier.oob_score_))
    # featureImportance = permutation_importance(classifier, X_test, Y_test, n_repeats=30, random_state=0)
  
sampleX = shap.sample(X_train, 100)
    # exp = shap.TreeExplainer(classifier, sampleX)
    # shap_values = exp.shap_values(X_train)
exp = shap.TreeExplainer(classifier).shap_values(X_train)

pdb.set_trace()
f1 = plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar")
f1.savefig('/shap_summary_value.png', bbox_inches = 'tight', dpi=600)
plt.show()

f2 = plt.figure()
shap.summary_plot(shap_values, X_train)
f2.savefig('/shap_variable_importance.png', bbox_inches = 'tight', dpi=600)
plt.show()