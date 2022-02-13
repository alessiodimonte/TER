from flask import Flask
import json
import os

api = Flask(__name__)

@api.route('/start_prediction', methods=['GET'])
def startTraining():
    os.system("python ../maskRCNN_24012022_v3/finalNotebook_test.py") 
    response = json.dumps({'success':True}), 200, {'Access-Control-Allow-Origin' : '*'}
    return response 
    

if __name__ == "__main__":
    api.run()
