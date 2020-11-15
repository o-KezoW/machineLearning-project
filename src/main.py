import flask
import numpy as np
import testing.save_model
import flask_http_response

app = flask.Flask(__name__)


def predict_diabetes(filled_form):
    predict = np.array(filled_form).reshape(1, 8)
    predict = predict.astype("int32")
    saved_model = testing.save_model.load_file
    result = saved_model.predict(predict)
    result = result.tolist
    return result


@app.route("/")
def app_home():
    return flask.render_template("index.html")


@app.route("/result", methods=["POST"])
def app_result():
    if flask.request.method == "POST":
        form = flask.request.form.to_dict()
        form = list(form.values())
        # return flask_http_response.result.return_response(form)
        for i in form:
            int(i)
        form = list(map(predict_diabetes(form), form))
        result = predict_diabetes(form)  # TODO: Error in this line fix, kthx
        return flask_http_response.result.return_response(result)
        if int(result) == 1:
            predict = "Probably has diabetes"
        else:
            predict = "Probably doesn't have diabetes"

        # Returns our result to a html page
        return flask.render_template("result.html", predict=predict)


if __name__ == "__main__":
    app.run(debug=True)
