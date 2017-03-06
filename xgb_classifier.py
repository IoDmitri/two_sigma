import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb

data = pd.read_json("train.json")
test = pd.read_json("test.json")

params = {}
params["objective"] = "multi:softprob"
params["eta"] = 0.4
params["max_depth"] = 10
params["num_class"] = 3
params["eval_metric"] = "mlogloss"
#params['colsample_bytree'] = 0.7
params["silent"] = 1
params["min_child_weight"] =5 
num_rounds = 10
max_words=50

plist = list(params.items())

def assign_class(cls):
	return ["high", "medium", "low"].index(cls)

def count_features(features):
	return map(lambda s: s.lower().replace("-", "").replace("!", ""), features)


def train_xgb_classifier(x_train, y_train, x_test = None, y_test=None):
	d_train = xgb.DMatrix(x_train, label=y_train)
	e_list = [(d_train, "train")]
	if y_test is not None:
		d_valid = xgb.DMatrix(x_test, y_test)
		e_list.append((d_valid, "eval"))
		return xgb.train(plist, d_train, num_rounds, e_list, weight)
	else:
		return xgb.train(plist, d_train, num_rounds, e_list)

def generate_dataset(data, count_vectorizer, cv=False):
	columns_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id"]
	w_counts = data["features"].map(lambda x : count_vectorizer.transform(x).toarray().sum(axis=0))
	w_counts_df = pd.DataFrame([x.tolist() for x in w_counts], columns=count_vectorizer.get_feature_names(), index=data.index.values)
	x = pd.concat([data, w_counts_df], axis=1)
	columns_to_use.extend(count_vectorizer.get_feature_names())
	#x = data

	x["has_building_id"] = data["building_id"].map(lambda x: 1 if x == "0" else 0)
	columns_to_use.extend(["has_building_id"])
	y = None

	if cv:
		return train_test_split(x[columns_to_use], x["interest_level"].map(lambda x : assign_class(x)), random_state=5)

	if "interest_level" in data:
		y = data["interest_level"].map(lambda x : assign_class(x))

	x = x[columns_to_use]
	return x, y

def make_model(cv=False):
	vect = CountVectorizer(max_features=max_words, stop_words=["to"], ngram_range=(1,3))
	vect.fit(np.hstack(data["features"].values))

	x_train = None
	y_train = None
	x_test = None
	y_test = None

	if cv:
		x_train, x_test, y_train, y_test = generate_dataset(data, vect, cv=True)
	else:
		x_train, y_train = generate_dataset(data, vect)
		x_test, _ = generate_dataset(test, vect)

	listing_id_vals = x_test["listing_id"].values

	del x_train["listing_id"]
	del x_test["listing_id"]

	model = train_xgb_classifier(x_train, y_train)

	preds = model.predict(xgb.DMatrix(x_test))
	pred_df = pd.DataFrame(preds)
	pred_df.columns = ["high", "medium", "low"]
	pred_df["listing_id"] = listing_id_vals

	if cv:
		#print accuracy_score(y_test, map(lambda x: np.argmax(x), pred_df[["high", "medium", "low"]].values))
		print confusion_matrix(y_test, map(lambda x: np.argmax(x), pred_df[["high", "medium", "low"]].values))
	else:
		f_name = "prediction.csv"
		if os.path.exists(f_name):
			os.remove(f_name)
		pred_df[["listing_id", "high", "medium", "low"]].to_csv(f_name, index=False)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		make_model(cv=False)
	else:
		make_model(cv=True)