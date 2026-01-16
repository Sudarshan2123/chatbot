# import hashlib
# import base64
# # C:\Users\100744\Desktop\VAPT policy mentor\RBI-UI-TOTP\static\assets\css\bootstrap.min.css
# # Path to your app.js file
# file_path = "./static/assets/css/bootstrap.min.css"  # Replace with the actual path

# # Read the file content
# with open(file_path, "rb") as f:
#     script_content = f.read()

# # Compute SHA-256 hash
# sha256_hash = hashlib.sha256(script_content).digest()

# # Encode in base64
# base64_hash = base64.b64encode(sha256_hash).decode("utf-8")

# # Output the hash in CSP format
# print(f"'sha256-{base64_hash}'")



import hashlib
import base64

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        hash_obj = hashlib.sha256(content)
        return f"sha256-{base64.b64encode(hash_obj.digest()).decode()}"

print(get_file_hash('/home/postgres/RBI-UI-TOTP/static/app.js'))
# print(get_file_hash('./static/assets/plugins/global/plugins.bundle.js'))
# print(get_file_hash('./static/assets/js/scripts.bundle.js'))
# print(get_file_hash('./static/assets/plugins/custom/fullcalendar/fullcalendar.bundle.js'))
# print(get_file_hash('./static/assets/js/custom/widgets.js'))
# print(get_file_hash('./static/marked.min.js'))