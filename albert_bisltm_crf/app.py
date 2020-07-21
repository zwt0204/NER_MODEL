# -*- encoding: utf-8 -*-
"""
@File    : app.py
@Time    : 2020/7/21 17:15
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from flask import request, Flask
import json

APP = Flask(__name__)
from albert_bisltm_crf.predict import test
from datetime import datetime


@APP.route('/demo', methods=['GET', 'POST'])
def demo():
    data = str(request.data, encoding="utf-8").strip()
    data = json.loads(data)
    b = datetime.now()
    a = test.predict_batch(data)
    print(datetime.now() - b)
    result = a
    if result is not None:
        return str(result)
    else:
        return 'no data'


if __name__ == '__main__':
    APP.run(host='127.0.0.1', port=7777, debug=True)
