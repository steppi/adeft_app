import os
import pickle

from flask import (
    Blueprint, request, render_template, redirect, session, url_for
    )

bp = Blueprint('app', __name__)

@app.route('/')
def main():
    return render_template('index.jinja2')


@app.route('/', methods=['POST'])
def main_post():
    shortform = request.form['shortform']
    data_path = os.path.join('data', 'longforms',
                             f'{shortform}_longforms.pkl')
    try:
        with open(data_path, 'rb') as f:
            longforms = pickle.load(f)
    except EnvironmentError:
        return render_template('index.jinja2')
    longforms, scores = zip(*longforms)
    session['longforms'] = longforms
    session['scores'] = [round(score, 1) for score in scores]
    session['names'] = ['']*len(longforms)
    session['groundings'] = session['names'].copy()
    data = list(zip(longforms, scores, session['names'],
                    session['groundings']))
    return render_template('input.jinja2', data=data)


@app.route('/input', methods=['POST'])
def add_groundings():
    return render_template('input.jinja2', longforms=session['longforms'],
                           scores=session['scores'], names=session['names'],
                           groundings=session['groundings'])
