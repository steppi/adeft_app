import os
import json
import logging
from copy import deepcopy
from collections import defaultdict

from flask import (
    Blueprint, request, render_template, session, url_for, redirect
    )

from deft.modeling.classify import load_model

from .trips import trips_ground
from .locations import DATA_PATH
from .scripts.model_to_s3 import model_to_s3
from .scripts.consistency import (check_grounding_dict,
                                  check_model_consistency,
                                  check_consistency_grounding_dict_pos_labels,
                                  check_consistency_names_grounding_dict,
                                  check_names_consistency)


logger = logging.getLogger(__file__)

bp = Blueprint('ground', __name__)


@bp.route('/')
def main():
    return render_template('index.jinja2')


@bp.route('/longforms', methods=['POST'])
def load_longforms():
    shortform = request.form['shortform']
    session['shortform'] = shortform
    try:
        cutoff = float(request.form['cutoff'])
    except ValueError or TypeError:
        cutoff = 1.0
    try:
        data = _init_from_file(shortform)
    except ValueError:
        try:
            data = _init_with_trips(shortform, cutoff)
        except ValueError:
            return render_template('index.jinja2')
    (session['longforms'], session['scores'], session['names'],
     session['groundings'], session['pos_labels']) = data
    data, pos_labels = _process_data(*data)
    return render_template('input.jinja2', data=data, pos_labels=pos_labels)


@bp.route('/fix', methods=['POST'])
def fix():
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
    original_longforms = deepcopy(longforms)
    transition = {grounding: grounding for grounding, _ in longforms}
    transition['ungrounded'] = 'ungrounded'
    session['transition'] = transition
    session['model_name'] = model_name
    session['longforms'], session['names'] = longforms, names
    session['top_longforms'] = top_longforms
    session['original_longforms'] = original_longforms
    return render_template('fix.jinja2', longforms=longforms, names=names,
                           top_longforms=top_longforms)


@bp.route('/fix_groundings', methods=['POST'])
def fix_groundings():
    for key in request.form:
        if key.startswith('s.'):
            index = key.partition('.')[-1]
            new_name = request.form[f'new-name.{index}']
            new_ground = request.form[f'new-ground.{index}']
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
    return render_template('fix.jinja2', longforms=session['longforms'],
                           names=session['names'],
                           top_longforms=session['top_longforms'])


@bp.route('/submit_fix', methods=['POST'])
def submit_fix():
    model_name = session['model_name']
    models_path = os.path.join(DATA_PATH, 'models', model_name)
    with open(os.path.join(models_path,
                           f'{model_name}_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    model = load_model(os.path.join(models_path,
                                    f'{model_name}_model.gz'))
    transition = session['transition']
    names = session['names']
    grounding_dict = {shortform: {longform: transition[grounding]
                                  for longform, grounding in
                                  grounding_map.items()}
                      for shortform, grounding_map in grounding_dict.items()}

    if not check_grounding_dict(grounding_dict):
        logger.error('grounding dict has become inconsistent.\n'
                     'This should not happen if the program is working'
                     ' as expected.')
        return redirect(url_for('submit_fix'))

    for index, label in enumerate(model.estimator.classes_):
        model.estimator.classes_[index] = transition[label]

    with open(os.path.join(models_path,
                           f'{model_name}_pos_labels.json'), 'r') as f:
        pos_labels = json.load(f)
    pos_labels = sorted(transition[label] for label in pos_labels)

    if not check_consistency_grounding_dict_pos_labels(grounding_dict,
                                                       pos_labels):
        logger.error('pos labels exist that are not in grounding dict')
        return redirect(url_for('submit_fix'))

    if not check_consistency_names_grounding_dict(grounding_dict, names):
        logger.error('names have become out of sync with grounding dict.')
        return redirect(url_for('submit_fix'))

    if not check_model_consistency(model, grounding_dict, pos_labels):
        logger.error('Model state has become inconsistent.')
        return redirect(url_for('submit_fix'))

    # update groundings files created before training model
    groundings_path = os.path.join(DATA_PATH, 'groundings')
    names_dict = {}
    pos_labels_dict = {}
    for shortform, grounding_map in grounding_dict.items():
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_names.json'), 'r') as f:
            temp = json.load(f)
        names_dict[shortform] = {transition[label]: name
                                 for label, name in temp.items()}
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_pos_labels.json'), 'r') as f:
            temp = json.load(f)
        pos_labels_dict[shortform] = [transition[label]
                                      for label in pos_labels]

    if not check_names_consistency(names.values()):
        logger.error('Inconsistent names for equivalent shortforms.')
        return redirect(url_for('submit_fix'))

    all_pos_labels = sorted(pos_label for labels in pos_labels_dict.values()
                            for pos_label in labels)

    if not all_pos_labels == pos_labels:
        logger.error('positive labels have become out of sync for model'
                     ' and groundings files.')
        return redirect(url_for('submit_fix'))

    # update files for model
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

    # update groundings files used for training model
    for shortform, grounding_map in grounding_dict.items():
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_grounding_map.json'), 'w') as f:
            json.dump(grounding_map, f)
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_names.json'), 'w') as f:
            json.dump(names_dict[shortform], f)
        with open(os.path.join(groundings_path, shortform,
                               f'{shortform}_pos_labels.json'), 'w') as f:
            json.dump(pos_labels_dict[shortform], f)
    model_to_s3(model_name)
    session.clear()
    return render_template('index.jinja2')


@bp.route('/input', methods=['POST'])
def add_groundings():
    name = request.form['name']
    grounding = request.form['grounding']
    names, groundings = session['names'], session['groundings']
    if name and grounding:
        selected = request.form.getlist('select')
        for value in selected:
            index = int(value)-1
            names[index] = name
            groundings[index] = grounding
    session['names'], session['groundings'] = names, groundings
    session['pos_labels'] = list(set(session['pos_labels']) & set(groundings))
    data = (session['longforms'], session['scores'], session['names'],
            session['groundings'], session['pos_labels'])
    data, pos_labels = _process_data(*data)
    return render_template('input.jinja2', data=data, pos_labels=pos_labels)


@bp.route('/delete', methods=['POST'])
def delete_grounding():
    names, groundings = session['names'], session['groundings']
    for key in request.form:
        if key.startswith('delete.'):
            id_ = key.partition('.')[-1]
            index = int(id_) - 1
            names[index] = groundings[index] = ''
            break
    session['names'], session['groundings'] = names, groundings
    session['pos_labels'] = list(set(session['pos_labels']) & set(groundings))
    data = (session['longforms'], session['scores'], session['names'],
            session['groundings'], session['pos_labels'])
    data, pos_labels = _process_data(*data)
    session['pos_labels'] = pos_labels
    return render_template('input.jinja2', data=data, pos_labels=pos_labels)


@bp.route('/pos_label', methods=['POST'])
def add_positive():
    for key in request.form:
        if key.startswith('pos-label.'):
            label = key.partition('.')[-1]
            session['pos_labels'] = list(set(session['pos_labels']) ^
                                         set([label]))
            break
    data = (session['longforms'], session['scores'], session['names'],
            session['groundings'], session['pos_labels'])
    data, pos_labels = _process_data(*data)
    return render_template('input.jinja2', data=data, pos_labels=pos_labels)


@bp.route('/generate', methods=['POST'])
def generate_grounding_map():
    shortform = session['shortform']
    longforms = session['longforms']
    names = session['names']
    groundings = session['groundings']
    pos_labels = session['pos_labels']
    grounding_map = {longform: grounding if grounding else 'ungrounded'
                     for longform, grounding in zip(longforms, groundings)}
    names_map = {grounding: name for grounding, name in zip(groundings,
                                                            names)
                 if grounding and name}
    groundings_path = os.path.join(DATA_PATH, 'groundings', shortform)
    try:
        os.mkdir(groundings_path)
    except FileExistsError:
        pass
    with open(os.path.join(groundings_path,
                           f'{shortform}_grounding_map.json'), 'w') as f:
        json.dump(grounding_map, f)
    with open(os.path.join(groundings_path,
                           f'{shortform}_names.json'), 'w') as f:
        json.dump(names_map, f)
    with open(os.path.join(groundings_path,
                           f'{shortform}_pos_labels.json'), 'w') as f:
        json.dump(pos_labels, f)
    session.clear()
    return redirect(url_for('ground.main'))


def _init_with_trips(shortform, cutoff):
    longforms, scores = _load(shortform, cutoff)
    trips_groundings = [trips_ground(longform) for longform in longforms]
    names, groundings = zip(*trips_groundings)
    names = [name if name is not None else '' for name in names]
    groundings = [grounding if grounding is not None
                  else '' for grounding in groundings]
    labels = set(grounding for grounding in groundings if grounding)
    pos_labels = list(set(label for label in labels
                          if label.startswith('HGNC:') or
                          label.startswith('FPLX:')))
    return longforms, scores, names, groundings, pos_labels


def _init_from_file(shortform):
    longforms, scores = _load(shortform, 0)
    groundings_path = os.path.join(DATA_PATH, 'groundings', shortform)
    try:
        with open(os.path.join(groundings_path,
                               f'{shortform}_grounding_map.json'), 'r') as f:
            grounding_map = json.load(f)
        with open(os.path.join(groundings_path,
                               f'{shortform}_names.json'), 'r') as f:
            names = json.load(f)
        with open(os.path.join(groundings_path,
                               f'{shortform}_pos_labels.json'), 'r') as f:
            pos_labels = json.load(f)
    except EnvironmentError:
        raise ValueError
    groundings = [grounding_map.get(longform) for longform in longforms]
    groundings = ['' if grounding == 'ungrounded' else grounding
                  for grounding in groundings
                  if grounding is not None]
    names = [names.get(grounding) for grounding in groundings]
    names = [name if name is not None else '' for name in names]
    return longforms, scores, names, groundings, pos_labels


def _load(shortform, cutoff):
    longforms_path = os.path.join(DATA_PATH, 'longforms',
                                  f'{shortform}_longforms.json')
    try:
        with open(longforms_path, 'r') as f:
            scored_longforms = json.load(f)
    except EnvironmentError:
        raise ValueError(f'data not currently available for shortform'
                         '{shortform}')
    longforms, scores = zip(*[(longform, round(score, 1))
                              for longform, score in scored_longforms
                              if score > cutoff])
    return longforms, scores


def _process_data(longforms, scores, names, groundings, pos_labels):
    labels = sorted(set(grounding for grounding in groundings if grounding))
    labels.extend(['']*(len(longforms) - len(labels)))
    data = list(zip(longforms, scores, names, groundings, labels))
    return data, pos_labels
