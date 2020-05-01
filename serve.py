"""
Simple flask server for the interface
"""

import os
import json
import pandas as pd

from flask import Flask, request, redirect, url_for
from flask import render_template

# -----------------------------------------------------------------------------

app = Flask(__name__)

# Merge thirukkural files
df_1 = pd.read_csv('tamil_thirukkural_train.csv', encoding='utf-8')
df_2 = pd.read_csv('tamil_thirukkural_test.csv', encoding='utf-8')
df = pd.concat([df_1, df_2], axis=0)
df = df.reset_index()
x = df.to_dict(orient='records')
x = json.dumps(x, ensure_ascii=False)
jall = json.loads(x)

text = df['kural'] + ' ' + df['mk'] + ' ' + df['mv'] + ' ' + df['sp']

# load computed paper similarities
#with open('sim_tfidf_svm.json', 'r') as f:
    #sim_dict = json.load(f)

# load search dictionary for each paper
with open('search.json', 'r') as f:
    search_dict = json.load(f)

# OPTIONAL: load tweet dictionary, if twitter_daemon has run
'''tweets_dict = {}
if os.path.isfile('tweets.json'):
    with open('tweets.json', 'r') as f:
        tweets_dict = json.load(f)
# decorate each paper with tweets
for j in jall['rels']:
    j['tweets'] = tweets_dict.get(j['rel_doi'], [])
    j['tweets'].sort(key=lambda t: t['followers'], reverse=True) 

# do some precomputation since we're going to be doing lookups of doi -> doc index
doi_to_ix = {}
for i, j in enumerate(jall['rels']):
    doi_to_ix[j['rel_doi']] = i '''

# -----------------------------------------------------------------------------
# few helper functions for routes

def default_context(papers, **kwargs):
    """ build a default context for the frontend """
    gvars = {'num_papers': len(jall)}
    gvars.update(kwargs) # insert anything else from kwargs into global context
    context = {'papers': papers, 'gvars': gvars}
    return context

# -----------------------------------------------------------------------------
# routes below

@app.route("/search", methods=['GET'])
def search():
    q = request.args.get('q', '') # get the search request
    if not q:
        return redirect(url_for('main')) # if someone just hits enter with empty field

    qparts = q.lower().strip().split() # split by spaces

    # accumulate scores
    n = len(jall)
    scores = []
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in qparts)
        if score == 0:
            continue # no match whatsoever, dont include
        score += 1.0 * (n - i)/n # give a small boost to more recent papers (low index)
        scores.append((score, jall[i]))
    scores.sort(reverse=True, key=lambda x: x[0]) # descending
    papers = [x[1] for x in scores if x[0] > 0]
    if len(papers) > 20:
        papers = papers[:20]
    context = default_context(papers, sort_order='search', search_query=q)
    return render_template('index.html', **context)


@app.route('/')
def main():
    from numpy.random import randint
    val = randint(0, 1325,1)[0]
    papers = jall[val : val+5]
    context = default_context(papers, sort_order='latest')
    return render_template('index.html', **context)
