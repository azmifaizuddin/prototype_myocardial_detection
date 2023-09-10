from flask import Blueprint, render_template, url_for
from flask_login import login_required,  current_user
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, MultipleFileField, RadioField
from werkzeug.utils import secure_filename
from predict_mi import model_performance, create_confusion_matrix
import os

views = Blueprint('views', __name__)

class UploadFileForm(FlaskForm):
    file1 = FileField("File")
    model_choices = [('cnn', 'CNN'), ('lstm', 'LSTM'), ('dnn', 'DNN')]
    radio1 = RadioField('Pilih Model ', choices=model_choices)
    submit = SubmitField("Submit")

def is_csv_file(filename):
    return filename.lower().endswith('.csv')


@views.route('/',methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file_filenames = []
        filenames = []
        selected_model = form.radio1.data
        file1 = form.file1.data
        file1.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static/files', secure_filename(file1.filename)))
        file1_path =  'H:/PrototypeTA/website/static/files/'+ secure_filename(file1.filename)
        model_name = selected_model.upper()
        cf_matrix, performance_result, mean_performance_result = model_performance(file1_path, selected_model)
        matrix_name = 'matrix1.png'
        matrix_path = 'H:/PrototypeTA/website/static/files/' + matrix_name
        create_confusion_matrix(cf_matrix, matrix_path)
        
        return render_template("hasil_upload.html", cf_matrix = cf_matrix, result = performance_result, mean_result = mean_performance_result, user=current_user, matrix_path = matrix_path, model_name = model_name)

    return render_template("home.html", user=current_user, form=form)


