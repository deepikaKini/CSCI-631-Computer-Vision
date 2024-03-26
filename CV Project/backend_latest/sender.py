from flask import request,Flask, render_template,jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from draw import check
app = Flask(__name__)
CORS(app,resource={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

@app.route('/here')
def index():
    # print()
    points = check()
    print(points)
    data ={"data":"I am here","points":points}
    return data


if __name__ == '__main__':
    socketio.run(app, debug=True,port=5000)

