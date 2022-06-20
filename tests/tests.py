#!/usr/bin/env python3

from breakit.perturbers import Perturber

en_perturber = Perturber.get_perturber('en')

def test_english_negations():

    sent = 'I cannot understand why you could not show me.'

    # only separate negators are deleted
    neg_deleted = 'I cannot understand why you could show me.'
    assert en_perturber.delete_negate(sent, 'not') == neg_deleted

    # only separate negators are doubled
    neg_doubled = 'I cannot understand why you could not not show me.'
    assert en_perturber.double_negate(sent, 'not') == neg_doubled

    # no negator matches
    no_neg = 'I cannot understand why you could show me.'
    assert en_perturber.delete_negate(no_neg, 'not') == no_neg
    assert en_perturber.double_negate(no_neg, 'not') == no_neg

    sent2 = 'I do not understand why you could not show me.'

    # only first negator is deleted
    neg_deleted = 'I do understand why you could not show me.'
    assert en_perturber.delete_negate(sent2, 'not') == neg_deleted

    # only first negator is doubled
    neg_doubled = 'I do not not understand why you could not show me.'
    assert en_perturber.double_negate(sent2, 'not') == neg_doubled
