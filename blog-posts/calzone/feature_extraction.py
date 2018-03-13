''' Extractors for Reddit Titles '''
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from textblob import TextBlob
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain
from collections import Counter
import numpy as np
from textstat.textstat import textstat


class DataExtractor(BaseEstimator, TransformerMixin):
    """ Select Title """
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[['title']]


class ItemSelector(BaseEstimator, TransformerMixin):
    """ Select Feature """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class WordCount(BaseEstimator, TransformerMixin):
    """ Extract number of words in title"""

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        df = pd.DataFrame({'wordcount': titles.str.split().apply(len)})
        return df['wordcount'].values.reshape(-1, 1)


class CharCount(BaseEstimator, TransformerMixin):
    """ Extract number of words in title"""

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        df = pd.DataFrame({'charcount': titles.str.len()})
        return df['charcount'].values.reshape(-1, 1)


class Vowels(BaseEstimator, TransformerMixin):
    """ Extract number of words in title"""

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        df = pd.DataFrame({'vowels': titles.str.findall(r'(?i)([aeiou])').apply(len)})
        return df['vowels'].values.reshape(-1, 1)


class Consonants(BaseEstimator, TransformerMixin):
    """ Extract number of words in title"""

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        df = pd.DataFrame({'consonants': titles.str.findall(r'(?i)([^aeiou])').apply(len)})
        return df['consonants'].values.reshape(-1, 1)


class Polarity(BaseEstimator, TransformerMixin):
    """ Determine sentiment polarity """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        blobs = [TextBlob(sentence) for sentence in titles]
        polarity = [blob.sentiment.polarity for blob in blobs]
        df = pd.DataFrame({'polarity': polarity})
        return df.polarity.values.reshape(-1, 1)


class Subjectivity(BaseEstimator, TransformerMixin):
    """ Determine sentiment polarity """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        blobs = [TextBlob(sentence) for sentence in titles]
        subjectivity = [blob.sentiment.subjectivity for blob in blobs]
        df = pd.DataFrame({'subjectivity': subjectivity})
        return df.subjectivity.values.reshape(-1, 1)


class Nouns(BaseEstimator, TransformerMixin):
    """ Extract number of nouns in title """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        blobs = [TextBlob(sentence) for sentence in titles]
        noun_phrases = [len(blob.noun_phrases) for blob in blobs]
        df = pd.DataFrame({'noun_phrases': noun_phrases})
        return df.noun_phrases.values.reshape(-1, 1)


class Blob(BaseEstimator, TransformerMixin):
    """ Combine noun, subject, polar extractors """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        blobs = [(len(x.noun_phrases), x.sentiment.subjectivity, x.sentiment.polarity) for x in [TextBlob(sentence) for sentence in titles]]
        return blobs


class Polarity(BaseEstimator, TransformerMixin):
    """ polarity extraction """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        blobs = [(x.sentiment.polarity) for x in [TextBlob(sentence) for sentence in titles]]
        return blobs
    
    
class Words(BaseEstimator, TransformerMixin):
    """ Combine Vowels,Consonants, CharCount, and WordCount extractors """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        words = list(zip(titles.str.findall(r'(?i)([aeiou])').apply(len),
                         titles.str.findall(r'(?i)([^aeiou])').apply(len),
                         titles.str.len(),
                         titles.str.split().apply(len)))
        return words


class Readable(BaseEstimator, TransformerMixin):
    """ Extract scores related to sentence readability """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        words = list(zip(
                         [textstat.flesch_kincaid_grade(x) for x in titles],
                         [textstat.syllable_count(x) for x in titles],
                         [textstat.flesch_reading_ease(x) for x in titles]
                         )
                    )
        return words

class Exclude(BaseEstimator, TransformerMixin):
    """ Combine Vowels,Consonants, CharCount, and WordCount extractors """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        words = list(zip(titles.str.split().apply(len)))
        return words


class POS(BaseEstimator, TransformerMixin):
    """ Count the parts-of-speech present in each title """

    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        def add_pos_with_zero_counts(counter, keys_to_add):
            for k in keys_to_add:
                counter[k] = counter.get(k,0)
            return counter
        blobs = [x for x in [TextBlob(sentence).tags for sentence in titles]]
        possible_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                         'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS',
                         'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
                         'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                         'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']
        # possible_tags = sorted(set(list(zip(*chain(*blobs)))[1]))
        pos_counts = [Counter(list(zip(*x))[1]) for x in blobs]
        pos_counts_with_zero = [add_pos_with_zero_counts(x, possible_tags) for x in pos_counts]
        sent_vector = [[count for tag, count in sorted(x.most_common())] for x in pos_counts_with_zero]
        df = pd.DataFrame(sent_vector, columns=sorted(possible_tags))
        return df.values


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(str.strip(re.sub(r'[^\w\s]','',t))) for t in word_tokenize(articles)]
