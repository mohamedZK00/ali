from flask import Flask, render_template, request ,jsonify
import pickle
import numpy as np


app = Flask(__name__)

# تحميل النموذج المحفوظ مسبقاً
with open('GBRmodel87%.PLK', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def index():
    gdr  = int(request.args['gdr_1'])
    gdr_2 =int(request.args['gdr_2'])
    gdr_3 =int(request.args['gdr_3'])

    pred = model.predict(np.array([gdr,gdr_2,gdr_3]).reshape(1,-1))
    return jsonify(Student_Grade = str(pred))


if __name__ == '__main__':
    app.run(debug=True)
