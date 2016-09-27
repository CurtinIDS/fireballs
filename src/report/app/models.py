from app import db


class Fireball(db.Model):
    image = db.Column(db.String(64), primary_key=True)
    label = db.Column(db.String(64), index=True)    
    prediction = db.Column(db.String(64), index=True)
    confidence = db.Column(db.Float)

    def __repr__(self):
        return '<Fireball %d>' % (self.id)
