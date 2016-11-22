from flask import render_template
from sqlalchemy import func, and_

from app import app
from app import db
from app.models import Transient 


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/results/<view>')
def results(view):

    # Only show the errors / misclassifications
    if view == 'errors':
        transients = Transient.query.filter(Transient.label != Transient.prediction).all()
    # Show all classification results
    elif view == 'transients':
        transients = Transient.query.filter(Transient.prediction == 'transients').order_by(Transient.confidence.desc()).all()
    else:
        transients = Transient.query.all()

    return render_template(
        'results.html',
        view=view,
        transients=transients)
