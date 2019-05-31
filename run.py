from flask import Flask

from vynd_api.api.api_blueprint import api_bp

def create_app(config_filename):
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app
	
if __name__ == "__main__":
    app = create_app("config")
# host = '0.0.0.0' makes the server visible across the network (Extremly Visible Server)
    app.run(debug=True, threaded=True)