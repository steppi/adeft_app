import os
import json
import pandas as pd
from collections import Counter, defaultdict

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


from deft.recognize import DeftRecognizer
from deft.modeling.classify import DeftClassifier
from deft.modeling.corpora import DeftCorpusBuilder

from deft_app.locations import DATA_PATH
from deft_app.scripts.consistency import check_grounding_dict, \
    check_consistency_grounding_dict_pos_labels


def train(shortforms, additional=None, n_jobs=1):
    """Train a deft model and produce quality statistics"""
    if additional is None:
        additional = []
    # gather needed data
    groundings_path = os.path.join(DATA_PATH, 'groundings')
    texts_path = os.path.join(DATA_PATH, 'texts')
    models_path = os.path.join(DATA_PATH, 'models')

    grounding_dict = {}
    names = {}
    pos_labels = set()
    # combine grounding maps and names from multiple shortforms into one model
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

    if not check_grounding_dict(grounding_dict):
        raise RuntimeError('Inconsistent grounding maps for shortforms.')
    pos_labels = sorted(pos_labels)

    # model name is built up from shortforms in model
    # (most models only have one shortform)
    agg_name = ':'.join(sorted(shortforms))
    with open(os.path.join(texts_path, agg_name,
                           f'{agg_name}_texts.json'), 'r') as f:
        text_dict = json.load(f)
    with open(os.path.join(texts_path, agg_name,
                           f'{agg_name}_text_map.json'), 'r') as f:
        ref_dict = json.load(f)

    # get statistics for matches to standard patterns
    stats = deft_stats(grounding_dict, names, text_dict, ref_dict)

    # build corpus for training models
    refs, texts = zip(*text_dict.items())
    texts = [text for text in texts if text is not None]
    deft_cb = DeftCorpusBuilder(grounding_dict)
    corpus = deft_cb.build_from_texts(texts)

    # gather additional texts
    for grounding, name, agent_text in additional:
        names[grounding] = name
        with open(os.path.join(texts_path, agent_text,
                               f'{agent_text}_texts.json'), 'r') as f:
            additional_texts = json.load(f)
            corpus.extend([(text, grounding)
                           for text_ref, text in additional_texts.items()
                           if text_ref not in text_dict])
            pos_labels.append(grounding)

    pos_labels = sorted(set(pos_labels))
    if not check_consistency_grounding_dict_pos_labels(grounding_dict,
                                                       pos_labels):
        raise RuntimeError('Positive labels exist that are not in'
                           ' grounding dict.')

    train, labels = zip(*corpus)
    deft_cl = DeftClassifier(shortforms, pos_labels)
    params = {'C': [100.0], 'max_features': [10000],
              'ngram_range': [(1, 2)]}
    deft_cl.cv(train, labels, params, n_jobs=n_jobs, cv=5)
    cv = deft_cl.grid_search.cv_results_

    preds = cross_val_predict(deft_cl.estimator, train, labels, n_jobs=n_jobs,
                              cv=5)
    conf_matrix = confusion_matrix(labels, preds)
    cv_results = {'labels': sorted(set(labels)),
                  'conf_matrix': conf_matrix.tolist(),
                  'f1': {'mean': cv['mean_test_f1'][0],
                         'std': cv['std_test_f1'][0]},
                  'precision': {'mean': cv['mean_test_pr'][0],
                                'std': cv['std_test_pr'][0]},
                  'recall': {'mean': cv['mean_test_rc'][0],
                             'std': cv['std_test_rc'][0]}}

    logit = deft_cl.estimator.named_steps['logit']
    coef = logit.coef_
    classes = logit.classes_

    # calculate feature importance
    feature_names = deft_cl.estimator.named_steps['tfidf'].get_feature_names()
    important_terms = {}
    # when there are greater than 2 classes, the logistic regression model
    # will have a row of coefficients for each class. when there are only
    # two classes, there is only one row of coefficients
    if len(classes) > 2:
        for index, label in enumerate(classes):
            fi = pd.DataFrame({'name': feature_names,
                               'importance': coef[index, :]})
            fi.sort_values('importance', ascending=False, inplace=True)
            top = fi.head(20)
            bottom = fi.tail(20)
            important_terms[label] = {'top':
                                      list(zip(top['name'],
                                               top['importance'])),
                                      'bottom':
                                      list(zip(bottom['name'],
                                               bottom['importance']))}
    else:
        fi = pd.DataFrame({'name': feature_names,
                           'importance': coef})
        fi.sort_values('importance', ascending=False, inplace=True)
        top = fi.head(20)
        bottom = fi.tail(20)
        important_terms[classes[0]] = list(zip(top['name'],
                                               top['importance']))
        important_terms[classes[1]] = list(zip(bottom['name'],
                                               bottom['importance']))

    unlabeled = []
    recognizers = [DeftRecognizer(shortform, grounding_map)
                   for shortform, grounding_Map in grounding_dict.items()]
    for text in texts:
        for rec in recognizers:
            if rec.recognize(text):
                break
        else:
            unlabeled.append(text)

    preds = deft_cl.estimator.predict(texts)
    preds = dict(Counter(preds))
    data = {'stats': stats,
            'cv_results': cv_results,
            'preds_on_unlabeled': preds,
            'important_terms': important_terms}
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
    with open(os.path.join(models_path, agg_name,
                           f'{agg_name}_stats.json'), 'w') as f:
        json.dump(data, f)
    with open(os.path.join(models_path, agg_name,
                           f'{agg_name}_pos_labels.json'), 'w') as f:
        json.dump(pos_labels, f)
    return deft_cl


def deft_stats(grounding_dict, names_dict, text_dict, ref_dict):
    """Output deft pattern matching stats as dict that can jsonified"""
    # need to run each recognizer on every text
    recognizers = [DeftRecognizer(shortform, grounding_map)
                   for shortform, grounding_map in grounding_dict.items()]

    # given dict mapping stmt ids to text_ref ids, get dict mapping
    # text_ref ids to counts of stmts from those texts
    stmt_counts = defaultdict(int)
    for stmt, ref in ref_dict.items():
        stmt_counts[ref] += 1
    stmt_counts = dict(stmt_counts)

    # get all groundings from grounding_dict
    groundings = {value for grounding_map in grounding_dict.values()
                  for value in grounding_map.values()}
    # get stats in pandas dataframe. columns for groundings. 1 if
    # text has this grounding, 0 if it doesn't. Also stmt count for
    # each text_ref
    # start with row template as dict. build up row by row
    row_template = {grounding: 0
                    for grounding in groundings}
    row_template['text_ref'] = None
    row_template['num_stmts'] = 0
    df = []
    for ref, text in text_dict.items():
        row = row_template.copy()
        row['text_ref'] = ref
        row['num_stmts'] = stmt_counts[int(ref)]
        for recognizer in recognizers:
            ground = recognizer.recognize(text)
            for g in ground:
                row[g] = 1
        df.append(row)
    df = pd.DataFrame(df)

    # rearrange columns so text_ref is first and num_stmts is
    # last
    cols = df.columns.tolist()
    ref_col = cols.index('text_ref')
    cols = [cols[ref_col]] + cols[0:ref_col] + cols[ref_col+1:]
    nstmts_col = cols.index('num_stmts')
    cols = (cols[0:nstmts_col] + cols[nstmts_col+1:] +
            [cols[nstmts_col]])
    df = df[cols]

    output = {}
    # get shortforms for json
    output['shortforms'] = list(grounding_dict.keys())

    # get total number of statements and total texts
    nstmts = df.num_stmts.sum()
    ntexts = len(df)
    output['total'] = {'stmts': int(nstmts), 'texts': int(ntexts)}

    # get number of texts and matching defining pattern for
    # one of the shortforms and number of corresponding statements
    ground_cols = df.drop(['text_ref', 'num_stmts'], axis=1)
    matched = ground_cols.any(axis=1)

    output['match_pattern'] = {'stmts': int(df[matched].num_stmts.sum()),
                               'texts': int(matched.sum())}
    output['groundings'] = {}
    unique = df[~(ground_cols.sum(axis=1) > 1)]
    for column in unique.columns:
        if column in ['text_ref', 'num_stmts']:
            continue
        ntexts = unique[column].sum()
        nstmts = unique[unique[column] == 1]['num_stmts'].sum()
        output['groundings'][column] = {'stmts': int(nstmts),
                                        'texts': int(ntexts)}
    return output
