#imports Flask server from app package
from app import app
from app.inference import *

if __name__ == "__main__":
  #starts server on local machine listening on port defined in Flask configs.
  app.run(host="127.0.0.1", debug=app.config['DEBUG'], port=app.config["PORT"], threaded=True)