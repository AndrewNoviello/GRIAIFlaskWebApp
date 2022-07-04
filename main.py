import functools
import json
import os

import io
import flask
from flask import Flask, render_template, redirect, request
import base64
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

import google_auth, google_drive

app = Flask(__name__)
app.secret_key = 'griai'

app.register_blueprint(google_auth.app)
app.register_blueprint(google_drive.app)
FOLDER_ID = "1q4KZmKhoj3-ywqrC-uu4wzhxMzO9yTFo"
TARGET_FOLDER_ID = "1aLRP8fJYrTdPXLmyhfdSnxrMZg3F3nPl"

@app.route('/')
def index():
    if google_auth.is_logged_in():
        drive_fields = "files(id,name,mimeType,createdTime,modifiedTime,shared,webContentLink)"
        items = google_drive.build_drive_api_v3().files().list(
                        pageSize=20, orderBy="folder", q='trashed=false',
                        fields=drive_fields
                    ).execute()

        return flask.redirect('/dashboard') #return flask.render_template('list.html', files=items['files'], user_info=google_auth.get_user_info())

    return render_template('index.html')

@app.route('/idUnknown', methods=['POST'])
def idUnknown():  #ID Unknown species
    speciesNames = request.form.getlist("speciesName[]")
    speciesNums = request.form.getlist("speciesNum[]")
    image_id = request.form.get("fileId")

    ##Code to build out description string
    desc = ""
    for ct in range(len(speciesNames)):
        if speciesNames[ct] != "":
            desc += str(speciesNums[ct]) + " " + speciesNames[ct] + "\n"

    ##Code to move the file
    service = google_drive.build_drive_api_v3()
    src = FOLDER_ID
    trg = TARGET_FOLDER_ID

    # File's new metadata
    metadata = dict(description=desc)

    service.files().update(
        fileId=image_id,
        body=metadata,
        addParents=trg,
        removeParents=src
    ).execute()

    return redirect("/dashboard", code=303)

def read_classes():
    with open("static/classes.txt") as f:
        data = f.read().splitlines()
    return data

@app.route('/dashboard')
def dashboard():
    print("Fetching Dashboard")
    if google_auth.is_logged_in():
        drive_fields = "files(id,name,mimeType,createdTime,modifiedTime,shared,webContentLink)"
        # folderObj = google_drive.build_drive_api_v3().files().list(orderBy="folder", q="name='Test_GRI_Unknown_Src' and trashed=false", fields=drive_fields, pageSize=1000).execute()
        # folders = folderObj.get('files', [])  #Grabs element of the dictionary object returned by the Google Drive API
        # folderId = folders[0].get('id')  #Gets id of the first folder in the list - should only be one folder with the proper name
        folderId = FOLDER_ID
        files = google_drive.build_drive_api_v3().files().list(q="'" + folderId + "' in parents and trashed=false", fields=drive_fields, pageSize=1000).execute()
        print("Files returned: ", files)
        files = files.get('files', [])
        File_Id = files[0].get('id')
        request = google_drive.build_drive_api_v3().files().get_media(fileId=File_Id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0)
        im_data = base64.b64encode(fh.read()).decode("utf-8")

        speciesClasses = read_classes()

        return render_template("dashboard.html", img_data=im_data, num_files=len(files), species=speciesClasses, fileId=File_Id)

    return render_template('index.html')



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)