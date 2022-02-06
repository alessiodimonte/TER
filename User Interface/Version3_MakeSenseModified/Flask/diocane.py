from flask import Flask
import json
import os

api = Flask(__name__)

@api.route('/start_training', methods=['GET'])
def startTraining():
    os.system("jupyter notebook DeepFish_MaskRCNN.ipynb") 
    response = json.dumps({'success':True}), 200, {'Access-Control-Allow-Origin' : '*'}
    return response 
    

if __name__ == "__main__":
    api.run()
