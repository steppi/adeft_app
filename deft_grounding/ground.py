import os
import json
import pickle

from flask import (
    Blueprint, request, render_template, session, current_app, url_for,
    redirect
    )

from .trips import trips_ground


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
    longforms_path = os.path.join(current_app.config['DATA'], 'longforms',
                                  f'{shortform}_longforms.pkl')
    try:
        with open(longforms_path, 'rb') as f:
            longforms = pickle.load(f)
    except EnvironmentError:
        print(longforms_path)
        return render_template('index.jinja2')
    longforms, scores = zip(*[(longform, score)
                              for longform, score in longforms
                              if score > cutoff])
    session['longforms'] = longforms
    session['scores'] = [round(score, 1) for score in scores]
    guessed = [trips_ground(longform) for longform in longforms]
    names, groundings = zip(*guessed)
    names = [name if name is not None else '' for name in names]
    groundings = [grounding if grounding is not None
                  else '' for grounding in groundings]
    session['names'] = names
    session['groundings'] = groundings
    data = list(zip(session['longforms'], session['scores'],
                    session['names'], session['groundings']))
    return render_template('input.jinja2', data=data)


@bp.route('/input', methods=['POST'])
def add_groundings():
    name = request.form['name']
    grounding = request.form['grounding']
    if name and grounding:
        selected = request.form.getlist('select')
        names = session['names']
        groundings = session['groundings']
        for value in selected:
            index = int(value)-1
            names[index] = name
            groundings[index] = grounding
        session['names'] = names
        session['groundings'] = groundings
    data = list(zip(session['longforms'], session['scores'],
                    session['names'], session['groundings']))
    return render_template('input.jinja2', data=data)


@bp.route('/generate', methods=['POST'])
def generate_grounding_map():
    shortform = session['shortform']
    longforms = session['longforms']
    names = session['names']
    groundings = session['groundings']
    grounding_map = {longform: grounding if grounding else 'ungrounded'
                     for longform, grounding in zip(longforms, groundings)}
    names_map = {grounding: name for grounding, name in zip(groundings,
                                                            names)
                 if grounding and name}
    groundings_path = os.path.join(current_app.config['DATA'],
                                   'groundings', shortform)
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
    return redirect(url_for('ground.main'))
