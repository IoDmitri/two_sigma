import collections
import operator
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix , log_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from data_utils import *
from IdVectorizer import IdVectorizer

data = pd.read_json("train.json")
test = pd.read_json("test.json")

params = {}
params["objective"] = "multi:softprob"
params["eta"] = 0.6
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
early_stop = 1

plist = list(params.items())

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
	
def generate_dataset(data, count_vectorizer, disc_count_vectorizer, man_ids_vect, build_id_vect, cv=False):
	columns_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id", "photo_counts", "created_day"]

	w_counts = data["features"].map(lambda x : count_vectorizer.transform(x).toarray().sum(axis=0))
	w_counts_df = pd.DataFrame([x.tolist() for x in w_counts], columns=count_vectorizer.get_feature_names(), index=data.index.values)
	columns_to_use.extend(count_vectorizer.get_feature_names())

	d_counts = data["description"].map(lambda x: disc_count_vectorizer.transform([x]).toarray().sum(axis=0))
	d_c_feature_names = map(lambda x : "desc_" + x, disc_count_vectorizer.get_feature_names())
	d_counts_df = pd.DataFrame([x.tolist() for x in d_counts], columns=d_c_feature_names, index=data.index.values)
	columns_to_use.extend(d_c_feature_names)

	x = pd.concat([data, w_counts_df, d_counts_df], axis=1)

	x["photo_counts"] = x["photos"].map(lambda x: len(x))
	x["created"] = pd.to_datetime(x["created"])
	x["created_day"] = x["created"].dt.day
	
	x["has_building_id"] = data["building_id"].map(lambda x: 1 if x == "0" else 0)
	columns_to_use.extend(["has_building_id"])

	man_ids_vect.transform(x)
	columns_to_use.extend(man_ids_vect.get_feature_names())

	build_id_vect.transform(x)
	columns_to_use.extend(build_id_vect.get_feature_names())

	x = x[columns_to_use]

	#dummy:
	dummy = ["manager_id", "building_id"]
	x = pd.get_dummies(x, columns=dummy)

	y = None

	if cv:
		return train_test_split(x, data["interest_level"], random_state=5, stratify=y)

	if "interest_level" in data:
		y = data["interest_level"]

	return x, y

def make_model(data, test, cv=False):
	merged = pd.concat([data, test])
	f_c_vect = CountVectorizer(max_features=max_words, stop_words="english", ngram_range=(1,3))
	f_c_vect.fit(np.hstack(merged["features"].values))
	man_ids = IdVectorizer("manager_id")
	man_ids.fit(data, test, transform_interest=True)
	build_id_vect = IdVectorizer("building_id")
	build_id_vect.fit(data, test)
	d_vect = CountVectorizer(max_features=100, stop_words= "english", ngram_range=(1,3))
	d_vect.fit(np.hstack(data["description"].values))

	x_train = None
	y_train = None
	x_test = None
	y_test = None

	dummy = ["manager_id", "building_id"]
	
	x_train, x_test, y_train, y_test = generate_dataset(data, f_c_vect, d_vect, man_ids, build_id_vect, cv=True)

	del x_train["listing_id"]
	listing_id_vals = x_test["listing_id"].values
	del x_test["listing_id"]
	del merged

	model = train_xgb_classifier(x_train, y_train, x_test, y_test)

	if not cv:
		x_test, _ = generate_dataset(test, f_c_vect, d_vect, man_ids, build_id_vect)
		listing_id_vals = x_test["listing_id"].values
		del x_test["listing_id"]


	preds = model.predict(xgb.DMatrix(x_test, missing=-0.999))
	pred_df = pd.DataFrame(preds)
	pred_df.columns = ["high", "medium", "low"]
	pred_df["listing_id"] = listing_id_vals

	if cv:
		print "log-loss"
		print log_loss(y_test, preds)
		cm = confusion_matrix(y_test, map(lambda x: np.argmax(x), pred_df[["high", "medium", "low"]].values))
		plot_confusion_matrix(cm, ["high", "medium", "low"], True)
		s = sorted(model.get_fscore().items(),  key=operator.itemgetter(1), reverse=True)[0:10]
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