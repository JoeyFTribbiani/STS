from SingleSentenceFeatures import SingleSentenceFeatures
import csv
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import pickle
from nltk.corpus import stopwords

# model_path = "model.pkl"
# model = pickle.load(open(model_path, "rb"))
# ssf = SingleSentenceFeatures(model)
#
# with open("STS-2015.csv", "r") as file:
#     reader = csv.reader(file)
#     content = [row for row in reader]
#     texts = []
#     labels = []
#     for i in range(0, len(content), 2):
#         s1 = ' '.join([word for word in content[i][1].lower().split() if word not in stopwords.words('english')])
#         s2 = ' '.join([word for word in content[i+1][1].lower().split() if word not in stopwords.words('english')])
#         if s1.strip() != "" and s2.strip() != "":
#             texts += [s1, s2],
#             labels += float(content[i][0].split("#")[-1]),
# y_train = labels
# X_train_bow = ssf.bow_features_with_kernels(texts)
# X_train_bodt = ssf.bodt_features_with_kernels(texts)
# X_train_pwe = ssf.pwe_features_with_kernels(texts)
# X_all = np.asmatrix(np.concatenate((X_train_bow,X_train_bodt,X_train_pwe), axis=1)).astype(dtype=np.float64)
#
# train = {"x_bow":X_train_bow,
#          "x_bodt":X_train_bodt,
#          "x_pwe":X_train_pwe,
#          "x_all":X_all,
#          "y":y_train}
#
# pickle.dump(train,open("train.pkl", "wb"))

train = pickle.load(open("train.pkl", "rb"))
y_train = train["y"]
del train["y"]

param_grid={'learning_rate': [0.1],
            'max_depth':[4, 6],
            'loss':['ls','lad', 'huber', 'quantile'],
            'min_samples_leaf':[5, 9, 17],
            'max_features':[0.3, 0.1, 1]
            }
n_jobs = 4
clf_group = {}
for feature in train:
    clf = GradientBoostingRegressor(n_estimators=1000)
    gs_cv = GridSearchCV(clf, param_grid, n_jobs=4).fit(train[feature], y_train)
    print("Best Estimator Parameters for GB_"+feature[2:])
    print("---------------------------")
    print(gs_cv.best_params_)
    print("Train R-squared: %.2f" %gs_cv.score(train[feature],y_train))
    clf = GradientBoostingRegressor(n_estimators=1000, learning_rate=gs_cv.best_params_['learning_rate'],
                                    max_depth=gs_cv.best_params_['max_depth'],loss=gs_cv.best_params_['loss'],
                                    min_samples_leaf=gs_cv.best_params_['min_samples_leaf'],max_features=gs_cv.best_params_['max_features'])
    clf.fit(train[feature], y_train)
    clf_group["clf_"+feature[2:]] = clf

clf_path = 'clf.pkl'
pickle.dump(clf_group, open(clf_path, "wb"))
