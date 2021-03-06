import io
import tempfile

import flask

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import googleapiclient.discovery
from google_auth import build_credentials, get_user_info

from werkzeug.utils import secure_filename

app = flask.Blueprint('google_drive', __name__)


def build_drive_api_v3():
    credentials = build_credentials()
    return googleapiclient.discovery.build('drive', 'v3', credentials=credentials)

def save_image(file_name, mime_type, file_data):
    drive_api = build_drive_api_v3().files()

    generate_ids_result = drive_api.generateIds(count=1).execute()
    file_id = generate_ids_result['ids'][0]

    body = {
        'id': file_id,
        'name': file_name,
        'mimeType': mime_type,
    }

    media_body = MediaIoBaseUpload(file_data,
                                   mimetype=mime_type,
                                   resumable=True)

    drive_api.create(body=body,
                     media_body=media_body,
                     fields='id,name,mimeType,createdTime,modifiedTime').execute()

    return file_id


@app.route('/gdrive/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in flask.request.files:
        return flask.redirect('/dashboard')

    file = flask.request.files['file']
    if (not file):
        return flask.redirect('/dashboard')

    filename = secure_filename(file.filename)

    fp = tempfile.TemporaryFile()
    ch = file.read()
    fp.write(ch)
    fp.seek(0)

    mime_type = flask.request.headers['Content-Type']
    save_image(filename, mime_type, fp)

    return flask.redirect('/')

@app.route('/google/moveFile', methods=['GET', 'POST'])
def index():
    service = build_drive_api_v3()
    src = '17xMEyiGKCJ8m7T0oer1uVaaxohTY2XRq'
    trg = '1VlOuYNTQ4fYGKkXpJ4gK4R0cnjtCq8lr'
    query = f"parents = '{src}'"
    response = service.files().list(q=query).execute()
    files = response.get('files')
    nextPageToken = response.get('nextPageToken')

    while nextPageToken:
        response = service.files().list(q=query, pageToken=nextPageToken).execute()
        files.extend(response.get('files'))
        nextPageToken = response.get('nextPageToken')

    for f in files:
        if f['mimeType'] != 'application / vnd.google-apps.folder':
            service.files().update(
                fileId = f.get('id'),
                addParents = trg,
                removeParents = src
            ).execute()

    return flask.redirect('/')

@app.route('/viewIm/<file_id>', methods=['GET'])
def view_file(file_id):
    drive_api = build_drive_api_v3()

    metadata = drive_api.get(fields="name,mimeType", fileId=file_id).execute()

    request = drive_api.get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while done is False:
        status, done = downloader.next_chunk()

    fh.seek(0)

    return flask.send_file(
                     fh,
                     attachment_filename=metadata['name'],
                     mimetype=metadata['mimeType']
               )