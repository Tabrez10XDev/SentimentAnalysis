from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from detect import diagnose, features

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('features', location='args', action='append')


class Features(Resource):

    def get(self):
        return jsonify({'features': features()})


class Detect(Resource):

    def get(self):
        args = parser.parse_args()
        disease = diagnose(args["features"])
        return jsonify({
            "result": disease
        })


api.add_resource(Features, '/')
api.add_resource(Detect, '/detect')

# driver function
if __name__ == '__main__':
    app.run(debug=True)