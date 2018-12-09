from collections import Counter
from math import log
from typing import Hashable, Iterable


class TFIDF(object):
    def __init__(self):
        self.doc_word_freq = dict()  # type: dict[Hashable, Counter]

    def add_word(self, word, doc_name):
        """
        adds a word being counted

        :type word: Hashable
        :type doc_name: unicode
        :rtype: None
        """
        self.doc_word_freq.setdefault(doc_name, Counter())[word] += 1

    def remove_word(self, word, doc_name):
        """
        removes an word from the count
        does not check that count is non-negative

        :type word: Hashable
        :type doc_name: unicode
        :rtype: None
        """
        self.doc_word_freq.setdefault(doc_name, Counter())[word] -= 1

    def update(self, word_list, doc_name):
        """
        add a list of words to be counted

        :type word_list: Iterable[Hashable]
        :type doc_name: unicode
        :rtype: None
        """
        for word in word_list:
            self.add_word(word, doc_name)

    def generate_tfidf(self, tfidf_type=u'smooth'):
        """
        calculate tf-idf for all words in all documents

        :return: dict of dict of tfidf {doc_name: {word: score, ...}, ...}
        :rtype: dict[unicode, Counter[Hashable, float]]
        """

        # count doc frequency
        df = Counter()  # document frequency not dataframe
        for doc_name, word_freq in self.doc_word_freq.items():
            df.update(word_freq.keys())

        if tfidf_type == u'smooth':
            # inverse document frequency smooth
            total_docs = float(sum(df.values()))
            idf = {word: log(1 + total_docs / docs) for word, docs in df.items()}

            # log normalization for term frequency
            out = dict()
            for doc_name, word_freq in self.doc_word_freq.items():
                for word, freq in word_freq.items():
                    out.setdefault(doc_name, Counter())[word] = (1 + log(freq)) * idf[word]

        else:
            # no smoothing
            idf = {word: 1.0 / docs for word, docs in df.items()}

            # raw count for term frequency
            out = dict()
            for doc_name, word_freq in self.doc_word_freq.items():
                for word, freq in word_freq.items():
                    out.setdefault(doc_name, Counter())[word] = freq * idf[word]

        return out

    def generate_idf(self, tfidf_type=u'prob'):
        """
        calculate tf-idf for all words in all documents

        :return: dict of dict of tfidf {doc_name: {word: score, ...}, ...}
        :rtype: dict[unicode, Counter[Hashable, float]]
        """

        df = Counter()  # document frequency not dataframe
        for doc_name, word_freq in self.doc_word_freq.items():
            df.update(word_freq.keys())

        if tfidf_type == u'prob':
            # probabilistic inverse document frequency
            total_docs = sum(df.values())
            idf = {word: max(0.0, log((total_docs - docs) * 1.0 / docs)) for word, docs in df.items()}

        elif tfidf_type == u'smooth':
            # inverse document frequency smooth
            total_docs = float(sum(df.values()))
            idf = {word: log(1 + total_docs / docs) for word, docs in df.items()}

        else:
            # no smoothing
            idf = {word: 1.0 / docs for word, docs in df.items()}

        out = dict()
        for doc_name, word_freq in self.doc_word_freq.items():
            for word, freq in word_freq.items():
                out.setdefault(doc_name, Counter())[word] = idf[word]

        return out

    def generate_bm25(self, k1=1.2, b=0.75, d=0.0):
        """
        calculate okapi bm25 scores for all words in all documents
        at extreme values of the coefficient b, BM25 turns into BM15 (for b=0) and BM11 (for b=1)
        d (delta) is used as a tuning weight for offsetting doc length normalization (introduced in in BM25+)
        to get the score for a query of multiple terms, sum the scores for each term
        see also: https://en.wikipedia.org/wiki/Okapi_BM25
        see also: https://www.elastic.co/guide/en/elasticsearch/guide/current/pluggable-similarites.html

        :param k1: usually chosen in the range [1.2, 2.0]
        :type k1: float
        :param b: in the range of [0, 1]; usually chosen to be 0.75
        :type b: float
        :param d: can be set to 1.0 to offset doc length normalization
        :type d: float
        :return: dict of dict of bm25 {doc_name: {word: score, ...}, ...}
        :rtype: dict[unicode, Counter[Hashable, float]]
        """
        # no data
        if not len(self.doc_word_freq):
            return dict()

        # count document frequency and document lengths
        df = Counter()  # dataframe
        doc_len = Counter()
        for doc_name, word_freq in self.doc_word_freq.items():
            df.update(word_freq.keys())
            doc_len[doc_name] += sum(word_freq.values())

        # average document length
        avg_len = sum(doc_len.values()) * 1.0 / len(doc_len)

        # calculate inverse doc freq
        total_docs = sum(df.values())
        idf = {word: log((total_docs - docs + 0.5) / (docs + 0.5)) for word, docs in df.items()}

        # calculate score
        out = dict()
        for doc_name, word_freq in self.doc_word_freq.items():
            for word, freq in word_freq.items():
                out.setdefault(doc_name, Counter())[word] = \
                    idf[word] * ((freq * (k1 + 1)) * 1.0 / (freq + k1 * (1 - b + b * doc_len[doc_name] / avg_len)) + d)

        return out
