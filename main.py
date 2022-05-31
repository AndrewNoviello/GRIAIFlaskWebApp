import functools
import json
import os

import flask
from flask import Flask, render_template
from authlib.client import OAuth2Session
import google.oauth2.credentials
import googleapiclient.discovery

import google_auth

app = Flask(__name__)
app.secret_key = 'griai'

app.register_blueprint(google_auth.app)

@app.route('/')
def index():
    if google_auth.is_logged_in():
        user_info = google_auth.get_user_info()
        return render_template('home.html') #'<div>You are currently logged in as ' + user_info['given_name'] + '<div><pre>' + json.dumps(user_info, indent=4) + "</pre>"

    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()