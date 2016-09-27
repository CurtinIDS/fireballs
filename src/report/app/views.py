from flask import render_template
from sqlalchemy import func, and_

from app import app
from app import db
from app.models import Fireball 


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/results/<view>')
def results(view):

    # Only show the errors / misclassifications
    if view == 'errors':
        fireballs = Fireball.query.filter(Fireball.label != Fireball.prediction).all()
    # Show all classification results
    else:
        fireballs = Fireball.query.all()

    return render_template(
        'results.html',
        view=view,
        fireballs=fireballs)
