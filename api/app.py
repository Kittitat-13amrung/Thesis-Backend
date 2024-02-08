from  flask import Flask, request, flash, redirect
# from flask_cors import CORS, cross_origin
from model.prediction import guitar2Tab
import io
import json

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_audio = request.files.get('audio')
        if(file_audio):
            audio = io.BytesIO(file_audio.read())

            model = guitar2Tab()
            predictions = model.predict(audio)
            # app.logger.info(predictions[0])
            print(predictions)
            return "done"
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # # If the user does not select a file, the browser submits an
        # # empty file without a filename.
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('download_file', name=filename))

if (__name__ == '__main__'):
    app.run()