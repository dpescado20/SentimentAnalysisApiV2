from flask import Flask
from flask_restplus import Api, Resource
from classifier import Classifier

server = Flask(__name__)
app = Api(app=server)

name_space = app.namespace('analyzer', description='Social Sentiment Analyzer APIs')

cl = Classifier()   


@name_space.route("/predict<string:social_text>")
class MainClass(Resource):
    def get(self, social_text):
        cleaned = cl.clean_text(social_text)
        vectorized = cl.vectorize_text(cleaned)
        prediction = cl.predict_text(vectorized)
        return {
            "prediction": prediction
        }


if __name__ == '__main__':
    server.run(debug=True)
