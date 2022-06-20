#!/usr/bin/env python3

import sys
import re

from typing import Callable

import spacy
import pandas as pd


class Perturber(object):

    @staticmethod
    def get_perturber(lang: str) -> 'Perturber':
        if lang == 'en':
            return EnglishPerturber()
        elif lang == 'de':
            return GermanPerturber()
        else:
            raise NotImplementedError

    @staticmethod
    def _delete_negator(sentence: str, negator: str) -> str:
        return re.sub(f' {negator} ', ' ', sentence, count=1)

    @staticmethod
    def _double_negator(sentence: str, negator: str) -> str:
        return re.sub(f' {negator} ',
                      f' {negator} {negator} ',
                      sentence, count=1)

    def __map(self,
              series: pd.Series,
              method: Callable, name: str) -> pd.DataFrame:
        new = series['good-translation'].map(lambda x: method(x, self.negator))
        new = new.rename('incorrect-translation')
        df = pd.concat([series, new], axis=1)
        df['phenomena'] = name
        return df

    def __perturb_negation(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        if self.negator:
            subset = tsv_f[tsv_f['good-translation'].str.contains(self.negator)]

            deleted_negation_df = self.__map(subset,
                                             self._delete_negator,
                                             'deleted_negation')
            double_negation_df = self.__map(subset,
                                            self._double_negator,
                                            'double_negation')

            return pd.concat([deleted_negation_df, double_negation_df])

    def __call__(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Make all language-independent perturbations.
        '''
        perturb_methods = [getattr(self, method_name)
                           for method_name in dir(self)
                           if callable(getattr(self, method_name)) and
                           '__perturb' in method_name]

        adversarial_sets = []
        for method in perturb_methods:
            new_df = method(tsv_f)
            if type(new_df) == pd.DataFrame:
                adversarial_sets.append(new_df)

        return pd.concat(adversarial_sets)


class EnglishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'en'
        self.negator = 'not'
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('''Please install spacy model - en_core_web_sm - like this:
                  python -m spacy download en_core_web_sm''')
            sys.exit()


class GermanPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'de'
        self.negator = 'nicht'
        try:
            self.nlp = spacy.load('de_core_news_sm')
        except OSError:
            print('''Please install spacy model - de_core_news_sm - like this:
                  python -m spacy download de_core_news_sm''')
            sys.exit()
