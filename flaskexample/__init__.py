from flask import Flask
app= Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from flaskexample import views
