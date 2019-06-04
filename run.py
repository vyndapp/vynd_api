
import os

from flask import Flask

from setup import download_vggface_model
from vynd_api.api import settings
from vynd_api.api.api_blueprint import api_bp

def create_app(config_filename):
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

download_vggface_model()

settings.init_video_processor()
	
# if __name__ == "__main__":
app = create_app("config")
# port = int(os.environ.get('PORT', 5000))
# app.run(debug=True, host='127.0.0.1', threaded=True, port=port)