from flask import Flask
import os

from vynd_api.api.api_blueprint import api_bp

def create_app(config_filename):
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app
	
# if __name__ == "__main__":
app = create_app("config")
# port = int(os.environ.get('PORT', 5000))
# app.run(debug=True, host='0.0.0.0', threaded=True, port=port)