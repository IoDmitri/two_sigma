import itertools
import operator
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from data_utils import *
from IdVectorizer import IdVectorizer

data = pd.read_json("train.json")
test = pd.read_json("test.json")

params = {}
params["objective"] = "multi:softprob"
params["eta"] = 0.4
params["lambda"] = 0.3
params["max_depth"] = 35
params["max_delta_step"] = 5
params["num_class"] = 3
params["eval_metric"] = "mlogloss"
#params['colsample_bytree'] = 0.7
params["silent"] = 1
params["min_child_weight"] = 10
num_rounds = 15
max_words=20
early_stop = 10

plist = list(params.items())

# def custom_eval(preds, dtrain):
# 	label = dtrain.get_label()
# 	weight = dtrain.get_weight()
# 	weight = weight * float(test_size) / len(label)#refactorize with 55k test sample for CV
# 	s = sum( weight[i] for i in range(len(label)) if label[i] == 1.0 )
# 	b = sum( weight[i] for i in range(len(label)) if label[i] == 0.0 )
# 	ams = AMS(s,b)
# 	ds=(np.log(s/(b+10.)+1))/ams;
# 	db=(((b+10)*np.log(s/(b+10.)+1)-s)/(b+10.))/ams;
# 	preds = 1.0 / (1.0 + np.exp(-preds))#sigmod it
# 	grad = ds*(preds-label)+db*(1-(preds-label))
# 	hess = np.ones(preds.shape)/300. #constant
# 	return grad, hess

def count_features(features):
	return map(lambda s: s.lower().replace("-", "").replace("!", ""), features)


def train_xgb_classifier(x_train, y_train, x_test = None, y_test=None):
	d_train = xgb.DMatrix(x_train, label=y_train, missing=-0.999)
	e_list = [(d_train, "train")]
	if y_test is not None:
		d_valid = xgb.DMatrix(x_test, label=y_test)
		e_list.extend([(d_valid, "eval")])
		return xgb.train(plist, d_train, num_rounds, e_list, early_stopping_rounds=early_stop)
	else:
		return xgb.train(plist, d_train, num_rounds, e_list, early_stopping_rounds=early_stop)
	
def generate_dataset(data, count_vectorizer, man_ids_vect, build_id_vect, cv=False, dummy=None):
	columns_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id"]
	w_counts = data["features"].map(lambda x : count_vectorizer.transform(x).toarray().sum(axis=0))
	w_counts_df = pd.DataFrame([x.tolist() for x in w_counts], columns=count_vectorizer.get_feature_names(), index=data.index.values)
	x = pd.concat([data, w_counts_df], axis=1)
	columns_to_use.extend(count_vectorizer.get_feature_names())
	columns_to_use.extend(man_ids_vect.get_feature_names())
	columns_to_use.extend(build_id_vect.get_feature_names())

	x["has_building_id"] = data["building_id"].map(lambda x: 1 if x == "0" else 0)
	columns_to_use.extend(["has_building_id"])
	man_ids_vect.transform(x)
	build_id_vect.transform(x)
	x = x[columns_to_use]

	if dummy:
		x = pd.get_dummies(x, columns=dummy)

	y = None

	if cv:
		return train_test_split(x, data["interest_level"], random_state=5)

	if "interest_level" in data:
		y = data["interest_level"]

	return x, y

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    **lifted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html **
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2) 
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="green" if cm[i, j] > thresh else "blue")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_model(data, test, cv=False):
	merged = pd.concat([data, test])
	c_vect = CountVectorizer(max_features=max_words, stop_words=["to"], ngram_range=(1,3))
	c_vect.fit(np.hstack(merged["features"].values))
	man_ids = IdVectorizer("manager_id")
	man_ids.fit(data, test, transform_interest=True)
	build_id_vect = IdVectorizer("building_id")
	build_id_vect.fit(data, test)

	x_train = None
	y_train = None
	x_test = None
	y_test = None

	dummy = ["manager_id", "building_id"]
	if cv:
		x_train, x_test, y_train, y_test = generate_dataset(data, c_vect, man_ids, build_id_vect, cv=True, dummy= dummy)
	else:
		x_train, y_train = generate_dataset(data, c_vect, man_ids, build_id_vect, dummy= dummy)
		x_test, _ = generate_dataset(test, c_vect, man_ids, build_id_vect, dummy= dummy)

	listing_id_vals = x_test["listing_id"].values

	del x_train["listing_id"]
	del x_test["listing_id"]
	del merged

	model = train_xgb_classifier(x_train, y_train, x_test, y_test)
	preds = model.predict(xgb.DMatrix(x_test, missing=-0.999))
	pred_df = pd.DataFrame(preds)
	pred_df.columns = ["high", "medium", "low"]
	pred_df["listing_id"] = listing_id_vals

	if cv:
		print "Accuracy Score"
		print accuracy_score(y_test, map(lambda x: np.argmax(x), pred_df[["high", "medium", "low"]].values))
		cm = confusion_matrix(y_test, map(lambda x: np.argmax(x), pred_df[["high", "medium", "low"]].values))
		plot_confusion_matrix(cm, ["high", "medium", "low"], True)
		s = sorted(model.get_fscore().items(),  key=operator.itemgetter(1), reverse=True)
		print s
		plt.show()
	else:
		f_name = "prediction.csv"
		if os.path.exists(f_name):
			os.remove(f_name)
		pred_df[["listing_id", "high", "medium", "low"]].to_csv(f_name, index=False)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		make_model(data, test, cv=False)
	else:
		make_model(data, test, cv=True)