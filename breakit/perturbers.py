#!/usr/bin/env python3
import logging
import sys
import re

from collections import defaultdict
from typing import Callable, List, Tuple
import random

import spacy
import pint
import pandas as pd
import names

from quantulum3 import parser
from textdistance import levenshtein


random.seed(2022)
logger = logging.getLogger("perturbers")

# setup the unit registry for unit conversions
UREG = pint.UnitRegistry()

# define allowed choices for character and digit additions / substitutions
# note: for characters of non-ascii script we will choose a random character
# from the sentence context for simplicty.
LETTERS = list('abcdefghijklmnopqrstuvwxyz')
DIGITS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']



class Perturber(object):

    def __init__(self):
        self.method_dict = {'negation': self.__perturb_negation,
                            'units': self.__perturb_units,
                            'named-entities': self.__perturb_named_entities,
                            'numbers': self.__perturb_numbers,
                            'dates': self.__perturb_dates}
        self.nlp = None

    @staticmethod
    def get_perturber(lang: str) -> 'Perturber':
        langs = {'en': EnglishPerturber,
                 'de': GermanPerturber,
                 'fr': FrenchPerturber,
                 'es': SpanishPerturber,
                 'zh': ChinesePerturber,
                 'ja': JapanesePerturber,
                 'ko': KoreanPerturber,
                 'hr': CroatianPerturber,
                 'cz': CzechPerturber,
                 'da': DanishPerturber,
                 'nl': DutchPerturber,
                 'et': EstonianPerturber,
                 'hu': HungarianPerturber,
                 'lv': LatvianPerturber,
                 'lt': LithuanianPerturber,
                 'no': NorwegianPerturber,
                 'pl': PolishPerturber,
                 'pt': PortuguesePerturber,
                 'ro': RomanianPerturber,
                 'sk': SlovakPerturber,
                 'sl': SlovenianPerturber,
                 'sv': SwedishPerturber,
                 }
        if lang in langs:
            return langs[lang]()
        else:
            raise NotImplementedError

    @staticmethod
    def delete_negate(sentence: str, negator: str) -> str:
        '''
        Delete a negation token from a sentence.
        '''
        return re.sub(f' {negator} ', ' ', sentence, count=1)

    @staticmethod
    def double_negate(sentence: str, negator: str) -> str:
        '''
        Duplicate a negation token to create a double negation.
        '''
        return re.sub(f' {negator} ',
                      f' {negator} {negator} ',
                      sentence, count=1)

    def add_character_ne(self, sentence: str) -> str:
        '''
        Add a character in named entities, if no named entity occurs return None.
        '''
        # We only consider named entities of type person - correct tag
        # needs to be specified in the corresponding language class constructor
        tagged = self.nlp(sentence)
        nes = [ent.text for ent in tagged.ents if ent.label_ in self.ne_tags]
        if not nes:
            return None

        # Add a random character at a random position
        ne = random.choice(nes)
        pos = random.randint(0, len(ne)-1)

        # Ensure that we don't replace with the same character and if the
        # script is non-Latin choose another non-Latin character from the sentence
        if ne[pos] in LETTERS:
            letter = random.choice(LETTERS)
        else:
            letter = random.choice(list(set(list(sentence)) - set(LETTERS)))

        # Consider casing
        letter_cased = letter.upper() if ne[pos].isupper() else letter
        next = ne[pos].lower() if ne[pos].isupper() else ne[pos]
        try:
            return re.sub(ne, ne[:pos]+letter_cased+next+ne[pos+1:], sentence, count=1)
        except re.error:
            return None

    def delete_character_ne(self, sentence: str) -> str:
        '''
        Delete a character in named entities, if no named entity occurs return None.
        '''
        # We only consider named entities of type person - correct tag
        # needs to be specified in the corresponding language class constructor
        tagged = self.nlp(sentence)
        nes = [ent.text for ent in tagged.ents if ent.label_ in self.ne_tags]
        if not nes:
            return None

        # Delete a character at a random position
        ne = random.choice(nes)
        pos = random.randint(0, len(ne)-1)
        pos = pos + 1 if ne[pos] == ' ' else pos
        pos = pos + 1 if ne[pos].isupper() else pos
        try:
            return re.sub(ne, ne[:pos]+ne[pos+1:], sentence, count=1)
        except re.error:
            return None

    def substitute_character_ne(self, sentence: str) -> str:
        '''
        Substitute a character in named entities, if no named entity occurs return None.
        '''
        # We only consider named entities of type person - correct tag
        # needs to be specified in the corresponding language class constructor
        tagged = self.nlp(sentence)
        nes = [ent.text for ent in tagged.ents if ent.label_ in self.ne_tags]
        if not nes:
            return None

        # Substitute a character at a random position with a random character
        ne = random.choice(nes)
        pos = random.randint(0, len(ne)-1)
        pos = pos + 1 if ne[pos] == ' ' else pos

        # Ensure that we don't replace with the same character and if the
        # script is non-Latin choose another non-Latin character from the sentence
        if ne[pos] in LETTERS:
            letter = random.choice(list(set(LETTERS) - set([ne[pos]])))
        else:
            letter = random.choice(list(set(list(sentence)) - set([ne[pos]]) - set(LETTERS)))

        # Consider casing
        letter_cased = letter.upper() if ne[pos].isupper() else letter
        try:
            return re.sub(ne, ne[:pos]+letter_cased+ne[pos+1:], sentence, count=1)
        except re.error:
            return None

    def substitute_whole_ne(self, sentence: str) -> str:
        '''
        Substitute a whole named entity, if no named entity occurs return None.
        '''
        # We only consider named entities of type person - correct tag
        # needs to be specified in the corresponding language class constructor
        tagged = self.nlp(sentence)
        nes = [ent.text for ent in tagged.ents if ent.label_ in self.ne_tags]
        if not nes:
            return None

        # If named entity is a single name replace with a last name, if multiple
        # names replace with first and last name
        ne = random.choice(nes)
        if len(ne.split()) == 1:
            new_ne = names.get_last_name()
        else:
            new_ne = names.get_full_name()
        try:
            new_entity = re.sub(ne, new_ne, sentence, count=1)
        except re.error:
            return None
        return new_entity if new_entity != sentence else None

    @staticmethod
    def add_digit_number(sentence: str) -> str:
        '''
        Add a digit in a number, if no number occurs return None.
        '''
        # A simple regex lookup should be enough to cover digit-based numbers
        nums = re.findall(r'\d+', sentence)
        if not nums:
            return None

        # Add a random digit at a random position
        num = random.choice(nums)
        pos = random.randint(0, len(num)-1)
        to_ignore = set() if pos > 0 else set([0])
        digit = random.choice(list(set(DIGITS) - to_ignore))
        return re.sub(num, num[:pos]+digit+num[pos:], sentence, count=1)

    @staticmethod
    def delete_digit_number(sentence: str) -> str:
        '''
        Delete a digit in a number, if no number occurs return None.
        '''
        # A simple regex lookup should be enough to cover digit-based numbers
        nums = re.findall(r'\d+', sentence)
        if not nums:
            return None

        # Delete a digit at a random position
        num = random.choice(nums)
        num_digits = len(num)

        # We don't fully remove digits, so cannot delete a single digit
        # This will also protect cases like "5-year-long"
        if num_digits == 1:
            return None
        pos = random.randint(1, num_digits-1)
        return re.sub(num, num[:pos]+num[pos+1:], sentence, count=1)

    @staticmethod
    def substitute_digit_number(sentence: str) -> str:
        '''
        Substitute a digit in a number, if no number occurs return None.
        '''
        # A simple regex lookup should be enough to cover digit-based numbers
        nums = re.findall(r'\d+', sentence)
        if not nums:
            return None

        # Substitute a digit at a random position with a random digit
        num = random.choice(nums)
        pos = random.randint(0, len(num)-1)
        to_ignore = set([num[pos]]) if pos > 0 else set([num[pos], 0])
        digit = random.choice(list(set(DIGITS) - to_ignore))
        return re.sub(num, num[:pos]+digit+num[pos+1:], sentence, count=1)

    @staticmethod
    def substitute_whole_number(sentence: str) -> Tuple[List[str]]:
        '''
        Substitute a whole number, if no number occurs return None.
        '''
        # A simple regex lookup should be enough to cover digit-based numbers
        nums = re.findall(r'\d+', sentence)
        if not nums:
            return None

        # Substitute a full number with a random number from range 1 - 999
        num = random.choice(nums)
        new = str(random.randint(1, 999))
        new_num = re.sub(num, new, sentence, count=1)
        return new_num if new_num != sentence else None

    def change_units(self, sentence: str) -> str:
        '''
        Change values from one unit to another.
        '''
        if not sentence:
            return

        # Extract all unit mentions in sentence
        unit_toks = parser.parse(sentence)
        good_changes = []
        bad_changes = []
        change_types = []

        # Helper function to create the good and bad translation pairs
        def add_example(old_token, old_value, old_unit, new_value, new_unit, sentence):
            good_changes.append(re.sub(old_token, f'{new_value} {new_unit}', sentence))
            bad_changes.append(re.sub(old_token, f'{old_value} {new_unit}', sentence))
            change_types.append('unit-change-wrong-amount')

            good_changes.append(re.sub(old_token, f'{new_value} {new_unit}', sentence))
            bad_changes.append(re.sub(old_token, f'{new_value} {old_unit}', sentence))
            change_types.append('unit-change-wrong-unit')

        # Helper function that rounds to desired precision but returns
        # an integer if no fractional part.
        def match(value: float):
            value = round(value, 3)
            if str(value).endswith('.0'):
                value = int(value)
            return value

        for unit_tok in unit_toks:
            # Ignore these cases because quantulum3 will compute fractions
            if '-' in unit_tok.surface and 'and' in unit_tok.surface:
                continue

            # Temperature units need special handling
            if unit_tok.unit.name in ['degree Celsius', 'degree fahrenheit']:
                if unit_tok.unit.name == "degree fahrenheit":
                    unit_entity = UREG.Quantity(unit_tok.value, UREG.degF)
                    new_entity = unit_entity.to(UREG.degC)
                    old_unit = '°F'
                    new_unit = '°C'
                else:
                    unit_entity = UREG.Quantity(unit_tok.value, UREG.degC)
                    new_entity = unit_entity.to(UREG.degF)
                    old_unit = '°C'
                    new_unit = '°F'

                add_example(unit_tok.surface.strip('+'),
                            str(match(unit_tok.value)), old_unit,
                            str(match(new_entity.m)), new_unit)

            # Dozen needs special handling
            elif unit_tok.unit.name in ['dozen']:
                new_val = str(int(unit_tok.value * 12))
                add_example(unit_tok.surface.strip('+'),
                            str(match(unit_tok.value)), 'dozen',
                            new_val, '')

            # Pound needs special handling
            elif unit_tok.unit.name in ['pound-mass']:
                unit_entity = UREG.Quantity(unit_tok.value, UREG.pound)
                new_entity = unit_entity.to(UREG.kilograms)

                add_example(unit_tok.surface.strip('+'),
                            str(match(unit_tok.value)), 'pounds',
                            str(match(new_entity.m)), 'kilograms')

            elif unit_tok.unit.name in self.swaps:
                try:
                    unit_entity = UREG.parse_expression(f'{unit_tok.value} {unit_tok.unit.name}')
                    new_units = self.swaps[unit_tok.unit.name]
                    for new_unit in new_units:
                        new_entity = UREG.parse_expression(f'{unit_tok.value} {new_unit}')
                        new_entity = unit_entity.to(new_entity.u)

                        add_example(unit_tok.surface.strip('+'),
                                    str(match(unit_tok.value)), unit_tok.unit.name+'s',
                                    str(match(new_entity.m)), new_unit)
                except pint.errors.UndefinedUnitError:
                    pass
                except pint.errors.OffsetUnitCalculusError:
                    pass
                except pint.errors.DimensionalityError:
                    pass

        return good_changes, bad_changes, change_types

    def removed_double_units(self, sentence: str) -> str:
        '''
        Remove disambiguated values, e.g. 1km (1.6 miles).
        '''
        # Only consider sentences that contain units
        try:
            unit_toks = parser.parse(sentence)
        except AttributeError:
            return None

        allowed_units = set([u.unit.name for u in unit_toks]) - set(self.swaps)
        if len(allowed_units) == 0:
            return None

        # Remove units in brackets because those are repetitions of previous
        # units with other metrics
        elif len(unit_toks) > 1:
            for unit_tok in unit_toks[::-1]:
                l = unit_tok.span[0] - 1
                r = unit_tok.span[1]
                try:
                    if sentence[l] == '(' and sentence[r] == ')':
                        sentence = re.sub(fr'\ ?\({unit_tok.surface}\)\ ?', ' ', sentence)
                    elif sentence[l] == '(':
                        sentence = re.sub(fr'\ ?\({unit_tok.surface}', ' ', sentence)
                    elif sentence[r] == ')':
                        sentence = re.sub(fr'{unit_tok.surface}\)\ ?', ' ', sentence)
                except re.error:
                    pass
                except IndexError:
                    return None
        return sentence

    @staticmethod
    def abbreviate_months(sentence: str, months: dict) -> str:
        '''
        Abbreviate a month.
        '''
        for month in months:
            if month in sentence:
                if months[month]:
                    return re.sub(rf'{month}\.?', rf'{months[month]}', sentence, count=1)

    @staticmethod
    def change_months(sentence: str, months: dict) -> str:
        '''
        Replace a month with a different month name.
        '''
        for month in months:
            if month in sentence:
                new_month = random.choice([m for m in months if m not in [month]])
                return re.sub(month, new_month, sentence, count=1)

    @staticmethod
    def map(series: pd.Series,
              method: Callable,
              name: str,
              *args) -> pd.DataFrame:
        '''
        Create incorrect examples by applying a perturbation method / function
        to all rows in a Pandas series.
        '''
        new = series['good-translation'].map(lambda x: method(x, *args))
        new = new.rename('incorrect-translation')
        df = pd.concat([series, new], axis=1)
        df['phenomena'] = name
        return df

    def __perturb_negation(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply negation-related perturbations.
        '''
        if self.negator:
            # Extract the subset of good translations with negator tokens
            subset = tsv_f[tsv_f['good-translation'].str.contains(self.negator)]

            deleted_negation_df = self.map(subset,
                                           self.delete_negate,
                                           'deleted_negation',
                                           self.negator)
            double_negation_df = self.map(subset,
                                          self.double_negate,
                                          'double_negation',
                                          self.negator)
            return pd.concat([deleted_negation_df, double_negation_df])

    def __perturb_named_entities(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply named entity-related perturbations.
        '''
        def run_perturbations(df, type):
            add_character_ne = self.map(df,
                                        self.add_character_ne,
                                        f'{type}-add-character-ne')
            delete_character_ne = self.map(df,
                                           self.delete_character_ne,
                                           f'{type}-delete-character-ne')
            substitute_character_ne = self.map(df,
                                               self.substitute_character_ne,
                                               f'{type}-substitute-character-ne')
            substitute_whole_ne = self.map(df,
                                           self.substitute_whole_ne,
                                           f'{type}-substitute-whole-ne')
            return pd.concat([add_character_ne, delete_character_ne, substitute_character_ne, substitute_whole_ne])

        # Compute the levenshtein distance between translation and reference for surface similarity filtering
        tsv_f['lev-dist'] = tsv_f.apply(lambda x: levenshtein.distance(x['reference'],  x['good-translation']), axis=1)
        tsv_f = tsv_f.sort_values(by=['lev-dist'])

        # Level 1: random subset of data where NE changes will be appkied to translation
        sampled_mt = run_perturbations(tsv_f.sample(250), 'level1')

        # Levels 2 and 3: subset of most similar and least similar examples
        # respectively where NE changes will be applied to reference
        swapped = tsv_f.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})
        most_similar_ref = run_perturbations(swapped.head(250), 'level2')
        least_similar_ref = run_perturbations(swapped.tail(250), 'level3')
        swapped_most_similar_ref = most_similar_ref.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})
        swapped_least_similar_ref = least_similar_ref.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})

        return pd.concat([sampled_mt, swapped_most_similar_ref, swapped_least_similar_ref]).dropna()

    def __perturb_numbers(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply number-related perturbations.
        '''
        def run_perturbations(df, type):
            add_digit_number = self.map(df,
                                        self.add_digit_number,
                                        f'{type}-add-digit-number')
            delete_digit_number = self.map(df,
                                           self.delete_digit_number,
                                           f'{type}-delete-digit-number')
            substitute_digit_number = self.map(df,
                                               self.substitute_digit_number,
                                               f'{type}-substitute-digit-number')
            substitute_whole_number = self.map(df,
                                               self.substitute_whole_number,
                                               f'{type}-substitute-whole-number')
            return pd.concat([add_digit_number, delete_digit_number, substitute_digit_number, substitute_whole_number])

        # Compute the levenshtein distance between translation and reference for surface similarity filtering
        tsv_f['lev-dist'] = tsv_f.apply(lambda x: levenshtein.distance(x['reference'],  x['good-translation']), axis=1)
        tsv_f = tsv_f.sort_values(by=['lev-dist'])

        # Level 1: random subset of data where number changes will be appkied to translation
        sampled_mt = run_perturbations(tsv_f.sample(250), 'level1')

        # Levels 2 and 3: subset of most similar and least similar examples
        # respectively where number changes will be applied to reference
        swapped = tsv_f.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})
        most_similar_ref = run_perturbations(swapped.head(250), 'level2')
        least_similar_ref = run_perturbations(swapped.tail(250), 'level3')
        swapped_most_similar_ref = most_similar_ref.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})
        swapped_least_similar_ref = least_similar_ref.rename(columns={'reference': 'good-translation', 'good-translation': 'reference'})
        return pd.concat([sampled_mt, swapped_most_similar_ref, swapped_least_similar_ref]).dropna()

    def __perturb_units(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply units-related perturbations.
        '''
        # Remove already disambiguated units
        single_ref = tsv_f['reference'].map(lambda x: self.removed_double_units(x))
        single_hyp = tsv_f['good-translation'].map(lambda x: self.removed_double_units(x))
        units = pd.concat([tsv_f['source'], single_ref, single_hyp], axis=1)
        units = units.dropna()

        changed_units = defaultdict(list)
        for i, row in units.iterrows():
            unit_toks_hyp = sorted([(tok.value, tok.unit.name) for tok in parser.parse(row['good-translation'])])
            unit_toks_ref = sorted([(tok.value, tok.unit.name) for tok in parser.parse(row['reference'])])

            # Only consider examples where all units in translation match with reference
            if unit_toks_hyp != unit_toks_ref:
                continue

            # Get the perturbations and create the TSV examples
            good_hyps, bad_hyps, types = self.change_units(row['good-translation'])
            if good_hyps and bad_hyps:
                for g, b, t in zip(good_hyps, bad_hyps, types):
                    changed_units['source'].append(row['source'])
                    changed_units['reference'].append(row['reference'])
                    changed_units['good-translation'].append(g)
                    changed_units['incorrect-translation'].append(b)
                    changed_units['phenomena'].append(t)

        return pd.DataFrame(changed_units)

    def _eliminate_wrong_month_translations(self, row) -> bool:
        '''
        Check whether month in reference is also in good-translation:
        '''
        for month in self.months:
            if month in row['reference'] and month in row['good-translation']:
                return True
        return False

    def __perturb_dates(self, tsv_f: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply date-time related perturbations.
        '''
        tsv_f['matches'] = tsv_f.apply(lambda x: self._eliminate_wrong_month_translations(x), axis=1).dropna()

        abbreviated_month_names = self.map(tsv_f,
                                       self.abbreviate_months,
                                       'date-time',
                                       self.months)
        changed_month_names = self.map(tsv_f,
                                       self.change_months,
                                       'date-time',
                                       self.months)['incorrect-translation']
        abbreviated_month_names['good-translation'] = abbreviated_month_names['incorrect-translation']
        abbreviated_month_names['incorrect-translation'] = changed_month_names
        return abbreviated_month_names.dropna()

    def __call__(self, tsv_f: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
        '''
        Make all language-independent perturbations.
        '''
        # Get all perturbation methods
        if not methods:
            perturb_methods = [getattr(self, method_name)
                               for method_name in dir(self)
                               if callable(getattr(self, method_name)) and
                               '__perturb' in method_name]
        else:
            perturb_methods = [self.method_dict[m] for m in methods]

        # Apply all perturbation methods and combine the resulting data frames
        adversarial_sets = []
        for method in perturb_methods:
            logger.info(f"Applying {method.__name__}")
            new_df = method(tsv_f)
            if type(new_df) == pd.DataFrame:
                adversarial_sets.append(new_df)

        return pd.concat(adversarial_sets)


class EnglishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'en'
        self.negator = 'not'
        self.ne_tags = ['PERSON']
        self.swaps = {'mile': ['kilometres', 'metres'],
                      'kilometre': ['miles', 'metres'],
                      'metre': ['feet', 'yards'],
                      'foot': ['metres', 'yards'],
                      'centimetre': ['inches', 'millimetres'],
                      'inch': ['centimetres', 'millimetres'],
                      'millimetre': ['centimetres', 'inches'],
                      'mile per hour':  ['kilometres per hour'],
                      'kilometre per hour': ['miles per hour'],
                      'kilometre per second': ['miles per second'],
                      'mile per second': ['kilometres per second'],
                      'hour': ['minutes'],
                      'minute': ['seconds'],
                      'second': ['minutes'],
                      'day': ['hours'],
                      'month': ['weeks'],
                      'week': ['days'],
                      'barrel': ['gallons', 'litres'],
                      'gallon': ['barrels', 'litres'],
                      'kilogram': ['grams', 'pounds'],
                      'gram': ['ounces'],
                      'ounce': ['grams'],
                      'square kilometre': ['square miles']
                      }
        self.months = {'January': 'Jan.',
                       'February': 'Feb.',
                       'March': 'Mar.',
                       'April': 'Apr.',
                       'May': None,
                       'June': None,
                       'July': None,
                       'August': 'Aug.',
                       'September': 'Sept.',
                       'October': 'Oct.',
                       'November': 'Nov.',
                       'December': 'Dec.'}
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except OSError:
            print('''Please install spacy model - en_core_web_lg - like this:
                  python -m spacy download en_core_web_lg''')
            sys.exit()


class GermanPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'de'
        self.negator = 'nicht'
        self.ne_tags = ['PER']
        self.months = {'Januar': 'Jan.',
                       'Februar': 'Feb.',
                       'März': None,
                       'April': 'Apr.',
                       'Mai': None,
                       'Juni': None,
                       'Juli': None,
                       'August': 'Aug.',
                       'September': 'Sept.',
                       'Oktober': 'Okt.',
                       'November': 'Nov.',
                       'Dezember': 'Dez.'}
        try:
            self.nlp = spacy.load('de_core_news_lg')
        except OSError:
            print('''Please install spacy model - de_core_news_lg - like this:
                  python -m spacy download de_core_news_lg''')
            sys.exit()


class FrenchPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'fr'
        self.ne_tags = ['PER']
        self.months = {'janvier': 'janv.',
                       'février': 'févr.',
                       'mars': None,
                       'avril': None,
                       'mai': None,
                       'juin': None,
                       'juillet': 'juil.',
                       'août': None,
                       'septembre': 'sept.',
                       'octobre': 'oct.',
                       'novembre': 'nov.',
                       'décembre': 'déc.'}
        try:
            self.nlp = spacy.load('fr_core_news_lg')
        except OSError:
            print('''Please install spacy model - fr_core_news_lg - like this:
                  python -m spacy download fr_core_news_lg''')
            sys.exit()


class SpanishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'es'
        self.ne_tags = ['PER']
        self.months = {'enero': None,
                       'febrero': 'feb.',
                       'marzo': None,
                       'abril': 'abr.',
                       'mayo': None,
                       'junio': 'jun.',
                       'julio': 'jul.',
                       'agosto': None,
                       'septiembre': 'sept.',
                       'octubre': 'oct.',
                       'noviembre': 'nov.',
                       'diciembre': 'dic.'}
        try:
            self.nlp = spacy.load('es_core_news_lg')
        except OSError:
            print('''Please install spacy model - es_core_news_lg - like this:
                  python -m spacy download es_core_news_lg''')
            sys.exit()


class JapanesePerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'ja'
        self.ne_tags = ['PERSON']
        try:
            self.nlp = spacy.load('ja_core_news_lg')
        except OSError:
            print('''Please install spacy model - ja_core_news_lg - like this:
                  python -m spacy download ja_core_news_lg''')
            sys.exit()


class ChinesePerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'zh'
        self.ne_tags = ['PERSON']
        try:
            self.nlp = spacy.load('zh_core_web_lg')
        except OSError:
            print('''Please install spacy model - zh_core_web_lg - like this:
                  python -m spacy download zh_core_web_lg''')
            sys.exit()


class KoreanPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'ko'
        self.ne_tags = ['PS']
        try:
            self.nlp = spacy.load('ko_core_news_lg')
        except OSError:
            print('''Please install spacy model - ko_core_news_lg - like this:
                  python -m spacy download ko_core_news_lg''')
            sys.exit()


class CroatianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'hr'
        self.months = {'siječanj': 'sijec.',
                       'veljača': 'velj.',
                       'ožujak': 'ozuj.',
                       'travanj': 'trav.',
                       'svibanj': 'svib.',
                       'lipanj': 'lip.',
                       'srpanj': 'srp.',
                       'kolovoz': 'kol.',
                       'rujan': 'ruj.',
                       'listopad': 'list.',
                       'studeni': 'stud.',
                       'prosinac': 'pros.'}


class CzechPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'cz'
        self.months = {'leden': 'led.',
                       'únor': 'ún.',
                       'březen': 'brez.',
                       'duben': 'dub.',
                       'květen': 'kvet.',
                       'červen': 'cerv.',
                       'červenec': 'cerven.',
                       'srpen': 'srp.',
                       'září': 'zár.',
                       'říjen': 'ríj.',
                       'listopad': 'list.',
                       'prosinec': 'pros.'}


class DanishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'da'
        self.months = {'januar': 'jan.',
                       'februar': 'febr.',
                       'marts': None,
                       'april': None,
                       'maj': None,
                       'juni': None,
                       'juli': None,
                       'august': 'aug.',
                       'september': 'sept.',
                       'oktober': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}


class DutchPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'nl'
        self.months = {'januari': 'jan.',
                       'februari': 'feb.',
                       'maart': None,
                       'april': 'apr.',
                       'mei': None,
                       'juni': None,
                       'juli': None,
                       'augustus': 'aug.',
                       'september': 'sept.',
                       'oktober': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}


class EstonianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'et'
        self.months = {'jaanuar': 'jaan',
                       'veebruar': 'veebr',
                       'märts': None,
                       'aprill': 'apr',
                       'mai': None,
                       'juuni': None,
                       'juuli': None,
                       'august': 'aug',
                       'september': 'sept',
                       'oktoober': 'okt',
                       'november': 'nov',
                       'detsember': 'dets'}


class HungarianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'hu'
        self.months = {'január': 'jan.',
                       'február': 'feb.',
                       'március': 'márc.',
                       'április': 'ápr.',
                       'május': 'máj.',
                       'június': 'jun.',
                       'július': 'jul.',
                       'augusztus': 'aug.',
                       'szeptember': 'szept.',
                       'október': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}


class ItalianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'it'
        self.months = {'gennaio': 'genn.',
                       'febbraio': 'febbr.',
                       'marzo': 'mar.',
                       'aprile': 'apr.',
                       'maggio': 'magg.',
                       'giugno': None,
                       'luglio': None,
                       'agosto': 'ag.',
                       'settembre': 'sett.',
                       'ottobre': 'ott.',
                       'novembre': 'nov.',
                       'dicembre': 'dic.'}


class LatvianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'lv'
        self.months = {'janvāris': 'jan.',
                       'februāris': 'feb.',
                       'marts': None,
                       'aprīlis': 'apr.',
                       'maijs': None,
                       'jūnijs': None,
                       'jūlijs': None,
                       'augusts': 'aug.',
                       'septembris': 'sept.',
                       'oktobris': 'okt.',
                       'novembris': 'nov.',
                       'decembris': 'dec.'}


class LithuanianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'lt'
        self.months = {'sausis': 'saus.',
                       'vasaris': 'vas.',
                       'kovas': None,
                       'balandis': 'bal.',
                       'gegužė': 'geg.',
                       'birželis': None,
                       'liepa': None,
                       'rugpjūtis': 'rugp.',
                       'rugsėjis': 'rugs.',
                       'spalis': None,
                       'lapkritis': 'lapkr.',
                       'gruodis': 'gr.'}


class NorwegianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'no'
        self.months = {'januar': 'jan.',
                       'februar': 'febr.',
                       'mars': None,
                       'april': None,
                       'mai': None,
                       'juni': None,
                       'juli': None,
                       'august': 'aug.',
                       'september': 'sept.',
                       'oktober': 'okt.',
                       'november': 'nov.',
                       'desember': 'des.'}


class PolishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'pl'
        self.months = {'styczeń': 'stycz.',
                       'luty': None,
                       'marzec': 'mar.',
                       'kwiecień': 'kwiec.',
                       'maj': None,
                       'czerwiec': 'czerw.',
                       'lipiec': 'lip.',
                       'sierpień': 'sierp.',
                       'wrzesień': 'wrzes.',
                       'październik': 'pazdz.',
                       'listopad': 'listop.',
                       'grudzień': 'grudz.'}


class PortuguesePerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'pt'
        self.months = {'janeiro': 'jan.',
                       'fevereiro': 'fev.',
                       'março': None,
                       'abril': None,
                       'maio': None,
                       'junho': None,
                       'julho': None,
                       'agosto': None,
                       'setembro': 'set.',
                       'outubro': 'out.',
                       'novembro': 'nov.',
                       'dezembro': 'dez.'}


class RomanianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'ro'
        self.months = {'ianuarie': 'ian.',
                       'februarie': 'feb.',
                       'martie': 'mar.',
                       'aprilie': 'apr.',
                       'mai': None,
                       'iunie': None,
                       'iulie': None,
                       'august': 'aug.',
                       'septembrie': 'sept.',
                       'octombrie': 'oct.',
                       'noiembrie': 'noiem.',
                       'decembrie': 'dec.'}


class SlovakPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'sk'
        self.months = {'január': 'jan.',
                       'február': 'feb.',
                       'marec': 'mar.',
                       'apríl': 'apr.',
                       'máj': None,
                       'jún': None,
                       'júl': None,
                       'august': 'aug.',
                       'septembra': 'sept.',
                       'október': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}


class SlovenianPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'sl'
        self.months = {'januar': 'jan.',
                       'februar': 'feb.',
                       'marec': 'mar.',
                       'april': 'apr.',
                       'maj': None,
                       'junij': 'jun.',
                       'julij': 'jul.',
                       'avgust': 'avg.',
                       'september': 'sept.',
                       'oktober': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}


class SwedishPerturber(Perturber):

    def __init__(self):
        super().__init__()
        self.lang = 'sv'
        self.months = {'januari': 'jan.',
                       'februari': 'febr.',
                       'mars': None,
                       'april': None,
                       'maj': None,
                       'juni': None,
                       'juli': None,
                       'augusti': None,
                       'september': 'sept.',
                       'oktober': 'okt.',
                       'november': 'nov.',
                       'december': 'dec.'}
