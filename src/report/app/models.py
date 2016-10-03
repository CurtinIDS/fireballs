from app import db


class Transient(db.Model):
    image = db.Column(db.String(64), primary_key=True)
    label = db.Column(db.String(64), index=True)    
    prediction = db.Column(db.String(64), index=True)
    confidence = db.Column(db.Float)

    def __repr__(self):
        return '<Transient %d>' % (self.id)
