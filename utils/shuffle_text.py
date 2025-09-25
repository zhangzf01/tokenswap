## code borrowed from https://github.com/mertyg/vision-language-models-are-bows/blob/main/dataset_zoo/perturbations.py

import numpy as np
import spacy
import subprocess
import benepar
from zss import simple_distance, Node
from nltk import Tree
from typing import Dict, List, Tuple

# try:
nlp = spacy.load("en_core_web_sm")
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "nltk_data/models/benepar_en3"})
import json

def swap_nsubj_dobj(ex):
    doc = nlp(ex)
    ## find noun subject
    sub = obj = None
    swapped_word_mask = None
    for token in doc:
        if token.dep_ == 'nsubj':
            sub = token
        elif token.dep_ == 'dobj':
            obj = token
    if sub and obj:
        tokens = []
        swapped_word_mask = []
        for token in doc:
            if token.dep_ == 'nsubj':
                tokens.append(obj.text)
                swapped_word_mask.append(1)
            elif token.dep_ == 'dobj':
                tokens.append(sub.text)
                swapped_word_mask.append(1)
            else:
                tokens.append(token.text)
                swapped_word_mask.append(0)
        return ' '.join(tokens), swapped_word_mask
    else:
        # print(f'Subject and Object are not found, return original text: {ex}')
        return ex, swapped_word_mask
        
def swap_nsubj_verb(ex):
    doc = nlp(ex)
    ## find noun subject
    sub = verb = None
    swapped_word_mask = None
    for token in doc:
        if token.dep_ == 'nsubj':
            sub = token
        elif token.pos_ == "VERB" and token.dep_ == "ROOT":
            verb = token
    if sub and verb:
        tokens = []
        swapped_word_mask = []
        for token in doc:
            if token.dep_ == 'nsubj':
                tokens.append(verb.text)
                swapped_word_mask.append(1)
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                tokens.append(sub.text)
                swapped_word_mask.append(1)
            else:
                tokens.append(token.text)
                swapped_word_mask.append(0)
        return ' '.join(tokens), swapped_word_mask
    else:
        # print(f'Subject and Object are not found, return original text: {ex}')
        return ex, swapped_word_mask

def swap_verb_dobj(ex):
    doc = nlp(ex)
    ## find noun subject
    verb = obj = None
    swapped_word_mask = None
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            verb = token
        elif token.dep_ == 'dobj':
            obj = token
    if verb and obj:
        tokens = []
        swapped_word_mask = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                tokens.append(obj.text)
                swapped_word_mask.append(1)
            elif token.dep_ == 'dobj':
                tokens.append(verb.text)
                swapped_word_mask.append(1)
            else:
                swapped_word_mask.append(0)
                tokens.append(token.text)
        return ' '.join(tokens), swapped_word_mask
    else:
        # print(f'Subject and Object are not found, return original text: {ex}')
        return ex, swapped_word_mask
