import hashlib
import base64
import requests

def generate_sri_hash(url):
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content
        digest = hashlib.sha384(content).digest()
        return f"sha384-{base64.b64encode(digest).decode()}"
    else:
        return f"Failed to fetch script: HTTP {response.status_code}"

# Usage
url = "https://cdn.jsdelivr.net/npm/marked/marked.min.js"
# hash_value = generate_sri_hash(url)
# print(hash_value)


import hashlib
import base64

def generate_sri_hash(filename):
    with open(filename, 'rb') as file:
        content = file.read()
        digest = hashlib.sha384(content).digest()
        return f"sha384-{base64.b64encode(digest).decode()}"

# Usage
hash_value = generate_sri_hash('C:\\Users\\100744\\Desktop\\GCP\\Mia_bot\\static\\app.js')
print(hash_value)

