#!/usr/bin/python
from __future__ import print_function
import sys, httplib2, os, datetime, io
from googleapiclient.http import MediaIoBaseDownload
from time import gmtime, strftime
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools
from datetime import date
#########################################################################
# Pre-reqs - 
# sudo pip install -I google-api-python-client==1.3.2 --ignore-installed six
# Authorize API https://developers.google.com/drive/v3/web/quickstart/python#step_1_turn_on_the_api_name
# Download the client_secret.json to same dir as this script
#########################################################################
#########################################################################
# Fixing OSX el capitan bug ->AttributeError: 'Module_six_moves_urllib_parse' object has no attribute 'urlencode'
os.environ["PYTHONPATH"] = "/Library/Python/2.7/site-packages"
#########################################################################

CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE="drive_api_token.json"
SCOPES = 'https://www.googleapis.com/auth/drive'
APPLICATION_NAME = 'Drive File API - Python'
OUTPUT_DIR=str(date.today())+"_drive_backup"

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

def get_credentials():
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, TOKEN_FILE)
    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def prepDest():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return True
    return False

def downloadFile(file_name, file_id, mimeType, service):
    request = service.files().get_media(fileId=file_id)
    if "application/vnd.google-apps" in mimeType:
        if "document" in mimeType:
            request = service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            file_name = file_name + ".docx"
        else: 
            request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
            file_name = file_name + ".pdf"
    #fh = io.FileIO(file_name, mode='wb')
    #downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
    print("Downloading -- " + file_name)
    response = request.execute()
    with open(os.path.join(OUTPUT_DIR, file_name), "wb") as wer:
        wer.write(response)
    #done = False
    #while done is False:
        #status, done = downloader.next_chunk()
        #print(float(status.progress()*100), done)

def listFiles(service):
    def getPage(pageTok):
        return service.files().list(q="mimeType != 'application/vnd.google-apps.folder'",
               pageSize=1000, pageToken=pageTok, fields="nextPageToken,files(id,name,mimeType)").execute()
    pT = ''; files=[]
    while pT is not None:
        results = getPage(pT)
        pT = results.get('nextPageToken')
        files = files + results.get('files', [])
    return files

def main():
    if prepDest():
        credentials = get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)
        for item in listFiles(service):
            downloadFile(item.get('name'), item.get('id'), item.get('mimeType'), service)
    else:
        print('Backup has already run today!')

if __name__ == '__main__':
    main()
