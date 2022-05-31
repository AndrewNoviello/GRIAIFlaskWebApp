import functools
import json
import os

import flask
from flask import Flask, render_template, redirect

import google_auth, google_drive

app = Flask(__name__)
app.secret_key = 'griai'

app.register_blueprint(google_auth.app)
app.register_blueprint(google_drive.app)


@app.route('/')
def index():
    if google_auth.is_logged_in():
        drive_fields = "files(id,name,mimeType,createdTime,modifiedTime,shared,webContentLink)"
        items = google_drive.build_drive_api_v3().files().list(
                        pageSize=20, orderBy="folder", q='trashed=false',
                        fields=drive_fields
                    ).execute()

        return flask.render_template('list.html', files=items['files'], user_info=google_auth.get_user_info())

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if google_auth.is_logged_in():
        creds = google_auth.build_credentials()
        return render_template("dashboard.html")
    return render_template('index.html')



if __name__ == '__main__':
    app.debug = True
    app.run()