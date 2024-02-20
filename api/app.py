from  flask import Flask, request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
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
@cross_origin()
def upload_file():
    if request.method == 'POST':
        file_audio = request.files.get('audio')
        if(file_audio):
            # file_
            audio = io.BytesIO(file_audio.read())
            # prediction model
            model = guitar2Tab()
            predictions = model.predict(audio)

            # save audio to audio folder
            with open(f'{os.getcwd()}/audio/{predictions}.wav', 'wb') as f:
                f.write(audio.getbuffer())

            

            # return predictions


            # s3 bucket
            s3 = boto3.client('s3')

            # s3 variables
            filename = f'{os.getcwd()}/output/{predictions}.xml'
            bucket_name = "thesis-bucket-2024"

            # upload xml file
            s3.upload_file(filename, bucket_name, f'xml/{predictions}.xml',
                        ExtraArgs={'ACL': 'public-read'})
            
            # # upload original audio file
            s3.upload_file(f'{os.getcwd()}/audio/{predictions}.wav', bucket_name, f'audio/{predictions}.wav',
                        ExtraArgs={'ACL': 'public-read'})
            
            # Delete the xml file from folder
            output_folder = f'{os.getcwd()}/output'
            audio_folder = f'{os.getcwd()}/audio'
            xml_file = f'{output_folder}/{predictions}.xml'
            audio_file = f'{audio_folder}/{predictions}.wav'
            os.remove(xml_file)
            os.remove(audio_file)


            return jsonify({
                "status": 201,
                "filename": predictions,
                "original_audio_url": f"https://{bucket_name}.s3.amazonaws.com/audio/{predictions}.wav",
                "url": f"https://{bucket_name}.s3.amazonaws.com/xml/{predictions}.xml"
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