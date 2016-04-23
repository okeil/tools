#!/usr/bin/python
import json, os, pickle, httplib2, io
import evernote.edam.userstore.constants as UserStoreConstants
import evernote.edam.type.ttypes as Types
from evernote.api.client import EvernoteClient
from evernote.edam.notestore.ttypes import NoteFilter, NotesMetadataResultSpec
from datetime import date

# Pre-reqs: pip install evernote 
# API key from https://dev.evernote.com/#apikey

os.environ["PYTHONPATH"] = "/Library/Python/2.7/site-packages"

CREDENTIALS_FILE=".evernote_creds.json"
LOCAL_TOKEN=".evernote_token.pkl"
OUTPUT_DIR=str(date.today())+"_evernote_backup"

def prepDest():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return True
    return True

# Helper function to turn query string parameters into a 
# source: https://gist.github.com/inkedmn
def parse_query_string(authorize_url):
    uargs = authorize_url.split('?')
    vals = {}
    if len(uargs) == 1:
        raise Exception('Invalid Authorization URL')
    for pair in uargs[1].split('&'):
        key, value = pair.split('=', 1)
        vals[key] = value
    return vals

class AuthToken(object):
    def __init__(self, token_list):
        self.oauth_token_list = token_list

def authenticate():
    def storeToken(auth_token):
        with open(LOCAL_TOKEN, 'wb') as output:
            pickle.dump(auth_token, output, pickle.HIGHEST_PROTOCOL)    

    def oauthFlow():
        with open(CREDENTIALS_FILE) as data_file:    
            data = json.load(data_file)
            client = EvernoteClient(
                consumer_key = data.get('consumer_key'),
                consumer_secret = data.get('consumer_secret'),
                sandbox=False
            )
        request_token = client.get_request_token('https://assetowl.com')
        print(request_token)
        print("Token expired, load in browser: " + client.get_authorize_url(request_token))
        print "Paste the URL after login here:"
        authurl = raw_input()
        vals = parse_query_string(authurl)
        auth_token=client.get_access_token(request_token['oauth_token'],request_token['oauth_token_secret'],vals['oauth_verifier'])
        storeToken(AuthToken(auth_token))
        return auth_token

    def storeToken(auth_token):
        with open(LOCAL_TOKEN, 'wb') as output:
            pickle.dump(auth_token, output, pickle.HIGHEST_PROTOCOL)    

    def getToken():
        store_token=""
        if os.path.isfile(LOCAL_TOKEN):
            with open(LOCAL_TOKEN, 'rb') as input:
              clientt = pickle.load(input)
            store_token=clientt.oauth_token_list
        return store_token

    try:
        client = EvernoteClient(token=getToken(),sandbox=False)
        userStore = client.get_user_store()
        user = userStore.getUser()
    except Exception as e:
        print(e)
        client = EvernoteClient(token=oauthFlow(),sandbox=False)
    return client

def listNotes(client):
    note_list=[]
    note_store = client.get_note_store()
    filter = NoteFilter()    
    filter.ascending = False
    spec = NotesMetadataResultSpec(includeTitle=True)
    spec.includeTitle = True
    notes = note_store.findNotesMetadata(client.token, filter, 0, 100, spec)
    for note in notes.notes:
        for resource in note_store.getNote(client.token, note.guid, False, False, True, False).resources:
            note_list.append([resource.attributes.fileName, resource.guid])
    return note_list


def downloadResources(web_prefix, res_array):
    for res in res_array:
        res_url = "%sres/%s" % (web_prefix, res[1])
        print("Downloading: " + res_url + " to " + OUTPUT_DIR + res[0])
        h = httplib2.Http(".cache")
        (resp_headers, content) = h.request(res_url, "POST",
                                        headers={'auth': DEV_TOKEN})
        with open(os.path.join(OUTPUT_DIR, res[0]), "wb") as wer:
            wer.write(content)

def main():
    if prepDest():
        client = authenticate()
        user_store=client.get_user_store()
        web_prefix = user_store.getPublicUserInfo(user_store.getUser().username).webApiUrlPrefix
        downloadResources(web_prefix, listNotes(client))

if __name__ == '__main__':
    main()
