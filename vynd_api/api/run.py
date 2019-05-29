
from flask import Flask

import sys

from .api_blueprint import api_bp

def create_app(config_filename):
    app = Flask(__name__)
    #app.config.from_object(config_filename)
    app.register_blueprint(api_bp, url_prefix = '/api')
    # from Model import db
    # db.init_app(app)
    return app
	
if __name__ == "__main__":
    app = create_app("config")
# host = '0.0.0.0' makes the server visible across the network (Extremly Visible Server)
    app.run(debug = True, threaded=True, host='127.0.0.5')