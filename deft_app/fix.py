import os
import json
import logging
from copy import deepcopy
from collections import defaultdict

from flask import (
    Blueprint, request, render_template, session, url_for, redirect
    )

from deft.modeling.classify import load_model

from .locations import DATA_PATH
from .scripts.consistency import (check_grounding_dict,
                                  check_model_consistency,
                                  check_consistency_grounding_dict_pos_labels,
                                  check_consistency_names_grounding_dict,
                                  check_names_consistency)


logger = logging.getLogger(__file__)

bp = Blueprint('fix', __name__)


@bp.route('/fix_init', methods=['POST'])
def initialize():
    model_name = request.form['modelname']
    if not model_name:
        return render_template('index.jinja2')
    models_path = os.path.join(DATA_PATH, 'models', model_name)
    with open(os.path.join(models_path,
                           model_name + '_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(models_path, model_name + '_names.json')) as f:
        names = json.load(f)
    longforms = defaultdict(list)
    longform_scores = defaultdict(int)
    for shortform, grounding_map in grounding_dict.items():
        with open(os.path.join(DATA_PATH, 'longforms',
                               f'{shortform}_longforms.json'), 'r') as f:
            lf_scores = json.load(f)
            for lf, score in lf_scores:
                longform_scores[lf] += score
        for longform, grounding in grounding_map.items():
            if grounding != 'ungrounded':
                longforms[grounding].append(longform)
    top_longforms = {grounding: max(longform_list,
                                    key=lambda x: longform_scores[x])
                     for grounding, longform_list in longforms.items()}
    longforms = [[grounding, '\n'.join(longform)] for grounding, longform
                 in longforms.items()]

    model = load_model(os.path.join(models_path, f'{model_name}_model.gz'))
    labels = model.estimator.named_steps['logit'].classes_.tolist()
    labels = [label for label in labels if label != 'ungrounded']
    pos_labels = model.pos_labels

    original_longforms = deepcopy(longforms)
    transition = {grounding: grounding for grounding, _ in longforms}
    transition['ungrounded'] = 'ungrounded'
    session['transition'] = transition
    session['model_name'] = model_name
    session['longforms'], session['names'] = longforms, names
    session['top_longforms'] = top_longforms
    session['original_longforms'] = original_longforms
    session['labels'] = labels
    session['pos_labels'] = pos_labels
    return render_template('fix.jinja2', longforms=longforms, names=names,
                           top_longforms=top_longforms, labels=labels,
                           pos_labels=pos_labels)


@bp.route('/fix_change_grounding', methods=['POST'])
def change_grounding():
    for key in request.form:
        if key.startswith('s.'):
            index = key.partition('.')[-1]
    new_name = request.form[f'new-name.{index}'].strip()
    new_ground = request.form[f'new-ground.{index}'].strip()
    names = session['names']
    longforms = session['longforms']
    original_longforms = session['original_longforms']
    old_ground = longforms[int(index)-1][0]
    origin_ground = original_longforms[int(index)-1][0]
    if new_name:
        names[old_ground] = new_name
    if new_ground:
        longforms[int(index)-1][0] = new_ground
        names[new_ground] = names.pop(old_ground)
        transition = session['transition']
        transition[origin_ground] = new_ground
        top_longforms = session['top_longforms']
        top_longforms[new_ground] = top_longforms.pop(old_ground)
        session['top_longforms'] = top_longforms
        session['longforms'], session['names'] = longforms, names
        labels = session['labels']
        labels = [new_ground if label == old_ground else label
                  for label in labels]
        pos_labels = session['pos_labels']
        pos_labels = [new_ground if label == old_ground else label
                      for label in pos_labels]
        session['labels'] = labels
        session['pos_labels'] = pos_labels
    return render_template('fix.jinja2', longforms=session['longforms'],
                           names=session['names'],
                           top_longforms=session['top_longforms'],
                           labels=session['labels'],
                           pos_labels=session['pos_labels'])


@bp.route('/fix_toggle_positive', methods=['POST'])
def toggle_positive():
    pos_labels = session['pos_labels']
    for key in request.form:
        if key.startswith('pos-label.'):
            label = key.partition('.')[-1]
            pos_labels = list(set(pos_labels) ^ set([label]))
            session['pos_labels'] = pos_labels
    return render_template('fix.jinja2', longforms=session['longforms'],
                           names=session['names'],
                           top_longforms=session['top_longforms'],
                           labels=session['labels'],
                           pos_labels=session['pos_labels'])


@bp.route('/fix_submit', methods=['POST'])
def submit():
    model_name = session['model_name']
    # load existing model files
    model, grounding_dict, _ = _load_model_files(model_name)

    # transition maps old groundings to new groundings
    transition = session['transition']
    new_grounding_dict = {shortform: {longform: transition[grounding]
                                      for longform, grounding in
                                      grounding_map.items()}
                          for shortform, grounding_map
                          in grounding_dict.items()}

    if not check_grounding_dict(new_grounding_dict):
        message = ('grounding_dict has become inconsistent.\n'
                   'This should not happen if the program is working'
                   ' as expected.')
        logger.error(message)
        return render_template('error.jinja2', message=message)

    for index, label in enumerate(model.estimator.classes_):
        model.estimator.classes_[index] = transition[label]

    new_pos_labels = session['pos_labels']
    new_names = session['names']

    # check consistency of newly generated files
    if not check_consistency_grounding_dict_pos_labels(grounding_dict,
                                                       new_pos_labels):
        message = 'pos labels exist that are not in grounding dict'
        logger.error(message)
        return render_template('error.jinja2', message=message)

    if not check_consistency_names_grounding_dict(grounding_dict, new_names):
        message = 'names have become out of sync with grounding dict.'
        logger.error(message)
        return render_template('error.jinja2', message=message)

    if not check_model_consistency(model, grounding_dict, new_pos_labels):
        message = 'Model state has become inconsistent'
        logger.error(message)
        return render_template('error.jinja2', message=message)


    # update groundings files created before training model
    groundings_path = os.path.join(DATA_PATH, 'groundings')
    names_dict = {}
    pos_labels_dict = {}
    for shortform, grounding_map in grounding_dict.items():
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_names.json'), 'r') as f:
            temp = json.load(f)
        names_dict[shortform] = {transition[label]:
                                 new_names[transition[label]]
                                 for label, name in temp.items()}
        labels = [transition[label] for label in grounding_map.values()
                  if label != 'ungrounded']
        pos_labels_dict[shortform] = list(set(labels) &
                                          set(new_pos_labels))

    if not check_names_consistency(names_dict.values()):
        message = 'Inconsistent names for equivalent shortforms.'
        logger.error(message)
        return render_template('error.jinja2', message=message)

    all_pos_labels = sorted(pos_label for labels in pos_labels_dict.values()
                            for pos_label in labels)
    if not all_pos_labels == set(new_pos_labels):
        message = ('positive labels have become out of sync in model'
                   ' and groundings files.')
        logger.error(message)
        return render_template('error.jinja2', message=message)

    _update_model_files(model_name, model, new_grounding_dict, new_names,
                        new_pos_labels)

    # update groundings files used for training model
    for shortform, grounding_map in new_grounding_dict.items():
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_grounding_map.json'), 'w') as f:
            json.dump(grounding_map, f)
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_names.json'), 'w') as f:
            json.dump(names_dict[shortform], f)
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_pos_labels.json'), 'w') as f:
            json.dump(pos_labels_dict[shortform], f)
    session.clear()
    return render_template('index.jinja2')


def _load_model_files(model_name):
    models_path = os.path.join(DATA_PATH, 'models', model_name)
    with open(os.path.join(models_path,
                           f'{model_name}_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(models_path,
                           f'{model_name}_names.json')) as f:
        names = json.load(f)
    model = load_model(os.path.join(models_path,
                                    f'{model_name}_model.gz'))
    return model, grounding_dict, names


def _update_model_files(model_name, model, grounding_dict, names, pos_labels):
    models_path = os.path.join(DATA_PATH, 'models', model_name)
    model.pos_labels = pos_labels
    with open(os.path.join(models_path,
                           f'{model_name}_grounding_dict.json'), 'w') as f:
        json.dump(grounding_dict, f)
    with open(os.path.join(models_path,
                           f'{model_name}_names.json'), 'w') as f:
        json.dump(names, f)
    with open(os.path.join(models_path,
                           f'{model_name}_pos_labels.json'), 'w') as f:
        json.dump(pos_labels, f)
    model.dump_model(os.path.join(models_path, f'{model_name}_model.gz'))
