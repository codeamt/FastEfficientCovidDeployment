from flask import Flask
app = Flask(__name__, static_url_path='/app/static')
app.config.from_object('app.settings')
from app import routes

