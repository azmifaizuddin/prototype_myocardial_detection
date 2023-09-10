from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] ='whatever'
    app.config['UPLOAD_FOLDER'] ='static/files'
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_NAME}"
    db.init_app(app)

    from .views import views
    
    app.register_blueprint(views, url_prefix='/')

    from .models import User

    with app.app_context():
        db.drop_all()
        db.create_all()
    
   

    return app

