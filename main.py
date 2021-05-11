from flask import Flask
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

f = open("config.json", "r")
config = json.loads(f.read())

@app.route("/")
def index():
    return "Hello World!"


@app.route("/services")
def servicesList():
    return config


if __name__ == "__main__":
    app.run(port=8080, debug=True)
