from  flask import Flask, request, flash, redirect, jsonify
# from flask_cors import CORS, cross_origin
from model.prediction import guitar2Tab
import io
import boto3
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    s3 = boto3.client('s3')
    filename = f'{os.getcwd()}/output/test.xml'
    bucket_name = "thesis-bucket-2024"
    s3.upload_file(filename, bucket_name, 'test.xml',
                   ExtraArgs={'ACL': 'public-read'})
    # response = s3.list_buckets()
    return jsonify({
        "status": 201,
        "url": f"https://{bucket_name}.s3.amazonaws.com/test.xml"
    }), 201


@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_audio = request.files.get('audio')
        if(file_audio):
            # file_
            audio = io.BytesIO(file_audio.read())
            filename = file_audio.filename
            # prediction model
            model = guitar2Tab()
            predictions = model.predict(audio)

            # s3 bucket
            s3 = boto3.client('s3')

            # s3 variables
            filename = f'{os.getcwd()}/output/{predictions}.xml'
            bucket_name = "thesis-bucket-2024"
            s3.upload_file(filename, bucket_name, f'{predictions}.xml',
                        ExtraArgs={'ACL': 'public-read'})
            
            return jsonify({
                "status": 201,
                "url": f"https://{bucket_name}.s3.amazonaws.com/{predictions}.xml"
            }), 201
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