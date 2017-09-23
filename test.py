import pickle
from nltk.corpus import stopwords
from SingleSentenceFeatures import SingleSentenceFeatures
from os import listdir
import codecs
import re
import numpy as np

model_path = "model.pkl"
model = pickle.load(open(model_path, "rb"))
ssf = SingleSentenceFeatures(model)
clf_group = pickle.load(open('clf.pkl','rb'))


path = "test"
for name in ["postediting","question-question","headlines","plagiarism","answer-answer"]:
    f = codecs.open(path + "/STS2016.input." + name + ".txt", "rb", "utf-8")
    testTexts = []
    for row in f:
        row = row
        s = row.split("	")
        s1 = re.sub(r'[^\w]', ' ', s[0])
        s2 = re.sub(r'[^\w]', ' ', s[1])
        s1 = " ".join([word for word in s1.lower().split() if word not in stopwords.words('english')])
        s2 = " ".join([word for word in s2.lower().split() if word not in stopwords.words('english')])
        testTexts += [s1, s2],

    X_test_bow = ssf.bow_features_with_kernels(testTexts)
    X_test_bodt = ssf.bodt_features_with_kernels(testTexts)
    X_test_pwe = ssf.bodt_features_with_kernels(testTexts)
    X_test_all = np.asmatrix(np.concatenate((X_test_bow,X_test_bodt,X_test_pwe), axis=1)).astype(dtype=np.float64)

    for m in ["bow","bodt","pwe","all"]:
        clf = clf_group["clf_"+m]
        if m == "bow":
            X_test = X_test_bow
        elif m == "bodt":
            X_test = X_test_bodt
        elif m == "pwe":
            X_test = X_test_pwe
        else:
            X_test = X_test_all
        y_test = clf.predict(X_test)

        with open("output/"+m+"/predict."+name+"", "w") as pre:
            for p in y_test:
                pre.write(str(p)+"\n")