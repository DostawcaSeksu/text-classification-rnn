import pandas as pd
import re
from gensim.models import Word2Vec
import json

def expand_keywords(model, seed_keywords, topn=10, threshold=0.5):
    seed_set = set(seed_keywords)
    expanded_set = set(seed_keywords)

    