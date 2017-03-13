import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_utils import *

class IdVectorizer(object):
	def __init__(self, id_, unk = "unk"):
		self._dict = {}
		self._encoder = LabelEncoder()
		self.unk = unk
		self.id_ = id_
		self._trained = False

	def get_feature_names(self):
		return [self.id_, self.id_+"avg"]

	def fit(self, train, test, thresh=10, transform_interest=False):
		train["in_train"] = 1
		test["in_test"] = 1
		merged = pd.concat([train, test])
		merged[["in_train", "in_test"]] = merged[["in_train", "in_test"]].fillna(0)
		ids = merged.groupby(self.id_)[["in_train", "in_test"]].sum()
		ids = ids[ids["in_train"] != 0]
		ids["total"] = ids["in_train"] + ids["in_test"]
		ids = ids.sort_values(by="total")
		self._encoder.fit(np.append(ids[ids["total"] >= thresh].index.values, self.unk))
		ids_to_use = set(self._encoder.classes_)
		if transform_interest:
			train["interest_level"] = train["interest_level"].map(lambda x : assign_class(x))
		interest_by_id = train.groupby(self.id_)["interest_level"].mean().to_dict()
		
		for k, v in interest_by_id.iteritems():
			if k in ids_to_use:
				self._dict[k] = v

		self._trained = True

	def transform(self, data):
		data[self.id_ + "avg"] = data[self.id_].map(lambda x: self.avg_for_id(x))
	
		id_set = set(self._encoder.classes_)
		data[self.id_] = self._encoder.transform(data[self.id_].map(lambda x: x if x in id_set else self.unk))

	def avg_for_id(self, id_):
		if self._trained:
			if id_ in self._dict:
				return self._dict[id_]
			return -1.0
		else:
			raise ValueError("Vectorizer has not been fitted")