from  flask import Flask, request, flash, redirect, jsonify, session, flash, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import io
from azure.storage.blob import BlobClient
import os
import config
from model.prediction import guitar2Tab
import pyodbc
import jwt
from datetime import datetime, timedelta
from functools import wraps
from assets.jwt import token_required
from models.User import User

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

conn = pyodbc.connect('DRIVER='+config.DB_DRIVER+';SERVER='+config.DB_URL+';PORT=1433;DATABASE='+config.DB_NAME+';UID='+config.DB_USERNAME+';PWD='+config.DB_PWD)
db_cursor = conn.cursor()


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
@token_required
def hello_world():
    port = config.PORT

    return jsonify(port), 200

# get songs list
@app.route('/songs')
@cross_origin()
def songs():
    data = []
    db_cursor.execute("SELECT * FROM songs")
    columns = [column[0] for column in db_cursor.description]
    for row in db_cursor.fetchall():
        data.append(dict(zip(columns, row)))
    db_cursor.close()
            
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

        filename_without_extension = secure_filename(os.path.splitext(file_audio.filename)[0])

        # COMPLETED: AUTO INCREMENT PRIMARY ID - MODIFIED THE DATABASE TABLE

        # prediction model
        prediction = predict_model(audio, filename=filename_without_extension)
        
        # write to database
        db_cursor.execute("INSERT INTO songs(name, filename, artist, bpm, key_signature, time_signature, duration, tuning, genre) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", filename_without_extension, prediction['filename'], "Unknown", prediction['bpm'], prediction['key'], prediction['time_signature'], prediction['duration'], prediction['tuning'], "Unknown")
        db_cursor.commit()
        db_cursor.close()

        # upload xml and audio file to azure blob storage
        blob_urls = upload_file_to_blob_storage_and_delete_files(filename_without_extension)

        # return 201 response
        return jsonify({
            "status": 201,
            "filename": prediction['filename'],
            "original_audio_url": blob_urls.get("audio_url"),
            "url": blob_urls.get("xml_url")
        }), 201

@app.route("/users/create", methods=["POST"])
@cross_origin()
def add_user():
    try:
        user = request.json
        if not user:
            return {
                "message": "Please provide user details",
                "data": None,
                "error": "Bad request"
            }, 400
        # validate input
        # is_validated = validate_user(**user)
        # if is_validated is not True:
        #     return dict(message='Invalid data', data=None, error=is_validated), 400
        
        user = User().create(**user)
        if not user:
            return {
                "message": "User already exists",
                "error": "Conflict",
                "data": None
            }, 409
        return {
            "message": "Successfully created new user",
            "data": user
        }, 201
    
    except Exception as e:
        return {
            "message": "Something went wrong",
            "error": str(e),
            "data": None
        }, 500

@app.route("/users/login", methods=["POST"])
@cross_origin()
def login():
    try:
        data = request.json
        if not data:
            return {
                "message": "Please provide user details",
                "data": None,
                "error": "Bad request"
            }, 400
        # validate input
        # is_validated = validate_email_and_password(data.get('email'), data.get('password'))
        # if is_validated is not True:
        #     return dict(message='Invalid data', data=None, error=is_validated), 400

        user = User().login(
            data["email"],
            data["password"]
        )

        if user:
            try:
                # token should expire after 24 hrs
                user["token"] = jwt.encode(
                    {"user_id": user["id"]},
                    config.JWT_SECRET_KEY,
                    algorithm="HS256"
                )
                return {
                    "message": "Successfully fetched auth token",
                    "data": user
                }
            except Exception as e:
                return {
                    "error": "Something went wrong",
                    "message": str(e)
                }, 500
        return {
            "message": "invalid email or password!",
            "data": None,
            "error": "Unauthorized"
        }, 404
    except Exception as e:
        return {
                "message": "Something went wrong!",
                "error": str(e),
                "data": None
        }, 500
    
@app.route('/logout')
@cross_origin()
def logout():
    session.pop('logged_in', None)
    return jsonify({'message': 'You are logged out'}), 200

if (__name__ == '__main__'):
    app.run(debug=True)



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