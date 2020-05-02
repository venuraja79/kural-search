"""
Run to update all the database json files that can be served from the website
"""

from tqdm import tqdm
import json
import requests
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------

def write_json(obj, filename, msg=''):
    suffix = f'; {msg}' if msg else ''
    print(f"writing {filename}{suffix}")
    with open(filename, 'w') as f:
        json.dump(obj, f)


def calculate_tfidf_features(rels, max_features=5000, max_df=1.0, min_df=3):
    """ compute tfidf features with scikit learn """
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        #token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b',
                        ngram_range=(1, 1), max_features=max_features,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=max_df, min_df=min_df)
    #corpus = [(a['rel_title'] + '. ' + a['rel_abs']) for a in rels]
    corpus = [(a) for a in rels]
    X = v.fit_transform(corpus)
    X = np.asarray(X.astype(np.float32).todense())
    print("tfidf calculated array of shape ", X.shape)
    return X, v


def calculate_sim_dot_product(X, ntake=40):
    """ take X (N,D) features and for each index return closest ntake indices via dot product """
    S = np.dot(X, X.T)
    IX = np.argsort(S, axis=1)[:, :-ntake-1:-1] # take last ntake sorted backwards
    return IX.tolist()


def calculate_sim_svm(X, ntake=40):
    """ take X (N,D) features and for each index return closest ntake indices using exemplar SVM """
    from sklearn import svm
    n, d = X.shape
    IX = np.zeros((n, ntake), dtype=np.int64)
    print(f"training {n} svms for each paper...")
    for i in tqdm(range(n)):
        # set all examples as negative except this one
        y = np.zeros(X.shape[0], dtype=np.float32)
        y[i] = 1
        # train an SVM
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-4, C=0.1)
        clf.fit(X, y)
        s = clf.decision_function(X)
        ix = np.argsort(s)[:-ntake-1:-1] # take last ntake sorted backwards
        IX[i] = ix
    return IX.tolist()


def build_search_index(rels, v):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # construct a reverse index for suppoorting search
    vocab = v.vocabulary_
    idf = v.idf_
    punc = "'!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'" # removed hyphen from string.punctuation
    trans_table = {ord(c): None for c in punc}

    def makedict(s, forceidf=None):
        words = set(s.lower().translate(trans_table).strip().split())
        words = set(w for w in words if len(w) > 1 and (not w in ENGLISH_STOP_WORDS))
        idfd = {}
        for w in words: # todo: if we're using bigrams in vocab then this won't search over them
            if forceidf is None:
                if w in vocab:
                    idfval = idf[vocab[w]] # we have a computed idf for this
                else:
                    idfval = 1.0 # some word we don't know; assume idf 1.0 (low)
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd

    def merge_dicts(dlist):
        m = {}
        for d in dlist:
            for k, v in d.items():
                m[k] = m.get(k,0) + v
        return m

    search_dict = []
    for p in rels:
        dict_title = makedict(p['kural'], forceidf=10)
        dict_adikaram = makedict(p['adikaram_name'], forceidf=5)
        dict_mk = makedict(p['mk'])
        dict_mv = makedict(p['mv'])
        dict_sp = makedict(p['sp'])
        qdict = merge_dicts([dict_title, dict_adikaram, dict_mk, dict_mv, dict_sp])
        #qdict = dict_summary
        search_dict.append(qdict)

    return search_dict


if __name__ == '__main__':

    # Merge thirukkural files
    df_1 = pd.read_csv('tamil_thirukkural_train.csv', encoding='utf-8')
    df_2 = pd.read_csv('tamil_thirukkural_test.csv', encoding='utf-8')
    df = pd.concat([df_1, df_2], axis=0)
    df = df.reset_index()
    x = df.to_dict(orient='records')
    x = json.dumps(x, ensure_ascii=False)
    jall = json.loads(x)

    text = df['kural'] + ' ' + df['mk'] + ' ' + df['mv'] + ' ' + df['sp']

    # calculate feature vectors for all abstracts and keep track of most similar other papers
    X, v = calculate_tfidf_features(text)

    # calculate the search index to support search
    search_dict = build_search_index(jall, v)
    write_json(search_dict, 'search.json')
