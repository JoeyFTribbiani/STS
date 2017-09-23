from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk import word_tokenize
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise
from nltk.parse.stanford import StanfordDependencyParser
from gensim import models

class SingleSentenceFeatures(object):

    def __init__(self, model=None):
        if not model:
            self.model = {}
        else:
            self.model = model

    def _tokenizer(self, text):
        words = word_tokenize(text)
        return words

    def _build_idf(self, texts):
        texts = [x for row in texts for x in row]
        vec = TfidfVectorizer(tokenizer=self._tokenizer, stop_words='english')
        s = vec.fit_transform(texts)
        self.model["idf"] = dict(zip(vec.get_feature_names(), enumerate(vec.idf_)))
        pickle.dump(self.model, open("model.pkl", "wb"))

    def bow_features(self, texts):
        # texts is an 2-D array object, each row has two sentences

        if "idf" not in self.model:
            self._build_idf(texts)

        d = []
        for (s1,s2) in texts:
            d += self._idf_vectorize(s1),
            d += self._idf_vectorize(s2),

        return d

    def _idf_vectorize(self, s):
        idf = self.model["idf"]
        vec = [0]*len(idf)
        for w in s:
            if w in idf:
                vec[idf[w][0]] = idf[w][1]
        return vec

    def bow_features_with_kernels(self, texts):
        d = np.asmatrix(self.bow_features(texts))
        feature_batch = []
        for i in range(0, len(d), 2):
            feature_batch += self.kernel_function(d[i], d[i + 1]),

        feature_batch = np.asmatrix(feature_batch).astype(dtype=np.float64)

        return feature_batch

    def _dependency_triple_analyzer(self, triples):
        return map(str, triples)

    def bodt_features(self, texts):
        path_to_jar = 'stanford-parser-full-2017-06-09/stanford-parser.jar'
        path_to_models_jar = 'stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
        dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

        t = [s for (s1, s2) in texts for s in (s1, s2)]
        dependency_triples = []

        # for s in t:
        #     print(2s)
        #     print(dependency_parser.raw_parse(s))

        for res in dependency_parser.raw_parse_sents(t):
            dependency_triples += next(res).triples(),

        if "bodt.tfidf" not in self.model:
            vec = TfidfVectorizer(lowercase=False, analyzer=self._dependency_triple_analyzer)
            s = vec.fit_transform(dependency_triples)
            self.model["bodt.tfidf"] = vec
            pickle.dump(self.model, open("model.pkl", "wb"))
        else:
            vec = self.model["bodt.tfidf"]
            s = vec.transform(dependency_triples)

        return s

    def bodt_features_with_kernels(self, texts):
        s = self.bodt_features(texts)
        d = s.todense()
        feature_batch = []
        for i in range(0, len(d), 2):
            feature_batch += self.kernel_function(d[i], d[i + 1]),

        feature_batch = np.asmatrix(feature_batch).astype(dtype=np.float64)

        return feature_batch

    def pwe_features(self, texts):
        m = models.KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        # if you vector file is in binary format, change to binary=True
        if "idf" not in self.model:
            self._build_idf(texts)

        idf = self.model["idf"]

        vectors = []

        i = 0
        for (s1, s2) in texts:
            wv1 = np.asarray([m[w]*idf[w][1] for w in s1 if w in m and w in idf])
            if wv1.size == 0:
                wv1_pool = [0]*900
            else:
                wv1_pool = np.concatenate((np.min(wv1, axis=0),np.max(wv1, axis=0),np.mean(wv1, axis=0)))
            wv2 = np.asarray([m[w]*idf[w][1] for w in s2 if w in m and w in idf])
            if wv2.size == 0:
                wv2_pool = [0]*900
            else:
                wv2_pool = np.concatenate((np.min(wv2, axis=0), np.max(wv2, axis=0), np.mean(wv2, axis=0)))
            vectors += wv1_pool, wv2_pool,

        return vectors

    def pwe_features_with_kernels(self, texts):
        d = np.asmatrix(self.pwe_features(texts))
        feature_batch = []
        for i in range(0, len(d), 2):
            feature_batch += self.kernel_function(d[i], d[i + 1]),

        feature_batch = np.asmatrix(feature_batch).astype(dtype=np.float64)

        return feature_batch

    def kernel_function(self, x1, x2):
        features = []

        # linear kernel:
        # Cosine distance
        features += np.squeeze(1 - pairwise.paired_cosine_distances(x1, x2)[0]),

        # Manhanttan distance
        features += pairwise.paired_manhattan_distances(x1, x2)[0],

        # Euclidean distance
        features += pairwise.paired_euclidean_distances(x1, x2)[0],

        # Chebyshev distance
        features += pairwise.pairwise_distances(x1, x2, metric="chebyshev")[0][0],

        # stat kernel:
        # Pearson coefficient
        pearson = stats.pearsonr(np.squeeze(np.asarray(x1)), np.squeeze(np.asarray(x2)))[0]
        features += 0 if np.isnan(pearson) else pearson,

        # Spearman coefficient
        spearman = stats.spearmanr(x1, x2, axis=1).correlation
        features += 0 if np.isnan(spearman) else spearman,

        # Kendall tau coefficient
        kendall = stats.kendalltau(x1, x2).correlation
        features += 0 if np.isnan(kendall) else kendall,

        # non-linear kernel:
        # polynomial
        features += pairwise.polynomial_kernel(x1, x2, degree=2)[0][0],

        # rbf
        features += pairwise.rbf_kernel(x1, x2)[0][0],

        # laplacian
        features += pairwise.laplacian_kernel(x1, x2)[0][0],

        # sigmoid
        features += pairwise.sigmoid_kernel(x1, x2)[0][0],

        return features
