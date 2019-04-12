import os
import json
import pickle
import argparse
import pandas as pd
from collections import defaultdict

from deft.recognize import DeftRecognizer
from deft.modeling.classify import DeftClassifier
from deft.modeling.corpora import DeftCorpusBuilder

from deft_app.locations import DATA_PATH


def train(shortforms, additional=None, n_jobs=1):
    groundings_path = os.path.join(DATA_PATH, 'groundings')
    texts_path = os.path.join(DATA_PATH, 'texts')
    models_path = os.path.join(DATA_PATH, 'models')

    grounding_dict = {}
    names = {}
    pos_labels = set()
    for shortform in shortforms:
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_grounding_map.json'), 'r') as f:
            grounding_map = json.load(f)
            grounding_dict[shortform] = grounding_map
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_names.json'), 'r') as f:
            names.update(json.load(f))
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_pos_labels.json'), 'r') as f:
            pos_labels.update(json.load(f))
    pos_labels = sorted(pos_labels)

    agg_name = ':'.join(sorted(shortforms))
    with open(os.path.join(texts_path, agg_name,
                           f'{agg_name}_texts.json'), 'r') as f:
        text_dict = json.load(f)
    with open(os.path.join(texts_path, agg_name,
                           f'{agg_name}_text_map.json'), 'r') as f:
        ref_dict = json.load(f)

    stats = deft_stats(grounding_dict, names, text_dict, ref_dict)
    return stats

    refs, texts = zip(*text_dict.items())
    texts = [text for text in texts if text is not None]
    deft_cb = DeftCorpusBuilder(grounding_dict)
    corpus = deft_cb.build_from_texts(texts)

    train, labels = zip(*corpus)
    deft_cl = DeftClassifier(shortforms, pos_labels)
    params = {'C': [100.0], 'max_features': [10000],
              'ngram_range': [(1, 2)]}
    deft_cl.cv(train, labels, params, n_jobs=n_jobs, cv=5)

    try:
        os.mkdir(os.path.join(models_path, agg_name))
    except FileExistsError:
        pass
    deft_cl.dump_model(os.path.join(models_path, agg_name,
                                    f'{agg_name}_model.gz'))
    with open(os.path.join(models_path, agg_name,
                           f'{agg_name}_grounding_dict.json'), 'w') as f:
        json.dump(grounding_dict, f)
    with open(os.path.join(models_path, agg_name,
                           f'{agg_name}_names.json'), 'w') as f:
        json.dump(names, f)
    return deft_cl, stats


def deft_stats(grounding_dict, names_dict, text_dict, ref_dict):
    recognizers = [DeftRecognizer(shortform, grounding_map)
                   for shortform, grounding_map in grounding_dict.items()]

    stmt_counts = defaultdict(int)
    for stmt, ref in ref_dict.items():
        stmt_counts[ref] += 1
    stmt_counts = dict(stmt_counts)
    groundings = {value for grounding_map in grounding_dict.values()
                  for value in grounding_map.values()}
    row_template = {grounding: 0
                    for grounding in groundings}
    row_template['text_ref'] = None
    row_template['num_stmts'] = 0
    out = []
    for ref, text in text_dict.items():
        row = row_template.copy()
        row['text_ref'] = ref
        row['num_stmts'] = stmt_counts[int(ref)]
        for recognizer in recognizers:
            ground = recognizer.recognize(text)
            for g in ground:
                row[g] = 1
        out.append(row)
    df = pd.DataFrame(out)
    cols = df.columns.tolist()
    ref_col = cols.index('text_ref')
    cols = [cols[ref_col]] + cols[0:ref_col] + cols[ref_col+1:]
    nstmts_col = cols.index('num_stmts')
    cols = cols[0:nstmts_col] + cols[nstmts_col+1:] + [cols[nstmts_col]]
    df = df[cols]

    nstmts = df.num_stmts.sum()
    ntexts = len(df)
    shortforms = list(grounding_dict.keys())
    shortforms = ' '.join(shortforms)
    out = f'For shortform {shortforms} there are:\n'
    out += f'{nstmts} statements from {ntexts} texts.\n'

    ground_cols = df.drop(['text_ref', 'num_stmts'], axis=1)
    matched = ground_cols.any(axis=1)
    out += (f'Of these texts, {matched.sum()} contain a defining pattern for'
            ' this shortform.\n')
    out += f'These account for {df[matched].num_stmts.sum()} statements.\n'
    unique = ground_cols[~(ground_cols.sum(axis=1) > 1)]
    for column in unique.columns:
        s = unique[column].sum()
        if column != 'ungrounded':
            out += (f'{s} texts can unambiguously be assigned grounding'
                    f' {column} with name {names_dict[column]}.\n')
        else:
            out += f'{s} texts have a pattern matching an ungrounded term.\n'
    print(out)
    return df
