from  flask import Flask, request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import io
from azure.storage.blob import BlobClient
import os
import config
from model.prediction import guitar2Tab
import pyodbc

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

cnxn = pyodbc.connect('DRIVER='+config.DB_DRIVER+';SERVER='+config.DB_URL+',1433;DATABASE='+config.DB_NAME+';UID='+config.DB_USERNAME+';PWD='+config.DB_PWD)

db_cursor = cnxn.cursor()

# check file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# upload file to blob storage then delete local xml and audio files
def upload_file_to_blob_storage_and_delete_files(filename):
    # get file from output folder
    xml_filepath = f'{os.getcwd()}/output/{filename}.xml'

    # upload xml and audio file to azure blob storage
    xml_blob = BlobClient.from_connection_string(
        conn_str=config.CONNECTION_STRING,
        container_name=config.CONTAINER_NAME,
        blob_name=f'xml/{filename}.xml'
    )

    # encode the file to binary and upload to blob storage
    with open(xml_filepath, "rb") as xml_file:
        xml_blob.upload_blob(xml_file)

    audio_filepath = f'{os.getcwd()}/audio/{filename}.wav'

    audio_blob = BlobClient.from_connection_string(
        conn_str=config.CONNECTION_STRING,
        container_name=config.CONTAINER_NAME,
        blob_name=f'audio/{filename}.wav'
    )

    with open(audio_filepath, "rb") as audio_file:
        audio_blob.upload_blob(audio_file)

    # Delete the xml file from folder
    output_folder = f'{os.getcwd()}/output'
    audio_folder = f'{os.getcwd()}/audio'
    xml_file = f'{output_folder}/{filename}.xml'
    audio_file = f'{audio_folder}/{filename}.wav'
    if os.path.exists(xml_file) and os.path.exists(audio_file):
        os.remove(xml_file)
        os.remove(audio_file)
    else:
        print("THe file does not exist")

    return { "audio_url": audio_blob.url, "xml_url": xml_blob.url }

# predict model function
def predict_model(file_audio, filename):
    # read audio file and convert to binary
    audio = io.BytesIO(file_audio.read())

    # run prediction model
    model = guitar2Tab()
    prediction = model.predict(audio, filename=filename)

    audio_filename = prediction['filename']

    # save audio to audio folder
    with open(f'{os.getcwd()}/audio/{audio_filename}.wav', 'wb') as f:
        f.write(audio.getbuffer())

    return prediction

# ROUTES
@app.route('/')
def hello_world():
    port = config.PORT

    return jsonify(port), 200

# get songs list
@app.route('/songs')
@cross_origin()
def songs():
    data = []

    with db_cursor as cursor:
        cursor.execute("SELECT * FROM songs")
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            data.append(dict(zip(columns, row)))
            
    return jsonify({"status": 200, "data": data}), 200

# predict model
@app.route('/predict', methods=['POST'])
@cross_origin()
def upload_file():
    # only allow post requests
    if request.method == 'POST':

        # check if the post request has the file part called 'audio'
        if 'audio' not in request.files:
            return jsonify({"status": 400, "message": "No file part"}), 400
        
        file_audio = request.files.get('audio')

        # check if the file extension is allowed
        if(file_audio and not allowed_file(file_audio.filename)):
            return jsonify({"status": 401, "message": "Incorrect file extension. Allowed file extensions are .wav, .mp3, and .ogg"}), 401

        # check if the post request has the file part called 'audio'
        audio = io.BytesIO(file_audio.read())

        # prediction model
        prediction = predict_model(audio, filename=os.path.splitext(file_audio.filename)[0])
        
        try:
            # write to database
            db_cursor.execute("INSERT INTO songs(name, filename, artist, bpm, key_signature, time_signature, duration, tuning, genre) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", file_audio.filename, prediction['filename'], "Unknown", prediction['bpm'], prediction['key'], prediction['time_signature'], prediction['duration'], prediction['tuning'], "Unknown")
            db_cursor.commit()

            # upload xml and audio file to azure blob storage
            blob_urls = upload_file_to_blob_storage_and_delete_files(os.path.splitext(file_audio.filename)[0])

            # return 201 response
            return jsonify({
                "status": 201,
                "filename": prediction['filename'],
                "original_audio_url": blob_urls.get("audio_url"),
                "url": blob_urls.get("xml_url")
            }), 201
        
        except Exception as e:
            return jsonify({"status": 500, "message": "An error occured while writing to the database", "errors": jsonify(e)}), 500

if (__name__ == '__main__'):
    app.run()



# # s3 bucket
# s3 = boto3.client('s3')

# # s3 variables
# bucket_name = "thesis-bucket-2024"

# # upload xml file
# s3.upload_file(filename, bucket_name, f'xml/{predictions}.xml',
#             ExtraArgs={'ACL': 'public-read'})

# # # upload original audio file
# s3.upload_file(f'{os.getcwd()}/audio/{predictions}.wav', bucket_name, f'audio/{predictions}.wav',
#             ExtraArgs={'ACL': 'public-read'})