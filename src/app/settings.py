"""
Onject Keys for Flask App Configuration.

"""

import os
PORT=5000
PATH="."
UPLOAD_FOLDER='app/static/uploads'
CONNECTED_NODE_ADDRESS="http://127.0.0.1:8000"
PYTORCH_FILES="app/models"
SECRET_KEY=os.urandom(24)