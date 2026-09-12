import json
import os
import uuid
import requests

username = 'verify' + str(uuid.uuid4()).split('-')[0]
payload = {'name': username, 'email': username + '@example.com', 'password': 'Passw0rd!'}
r = requests.post('http://127.0.0.1:8000/register', json=payload, timeout=10)
print('register', r.status_code)
print(r.text)
if r.ok:
    token = r.json().get('access_token')
    rr = requests.get('http://127.0.0.1:8000/history', headers={'Authorization': 'Bearer ' + token}, timeout=10)
    print('history', rr.status_code)
    print(rr.text[:1000])
