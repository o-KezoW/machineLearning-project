import flask
import numpy as np
import testing.save_model
# import flask_http_response

app = flask.Flask(__name__, template_folder="web/templates", static_folder="web/static")


def predict_diabetes(filled_form):
    predict = np.array(filled_form).reshape(1, -1).astype("float64")
    saved_model = testing.save_model.load_file
    result = saved_model.predict(predict)
    result = result.tolist()

    return result[0]


@app.route("/")
def app_home():
    return flask.render_template("index.html")


@app.route("/result", methods=["POST"])
def app_result():
    if flask.request.method == "POST":
        form = flask.request.form.to_dict()
        form = list(form.values())

        # form = list(map(predict_diabetes(form), form))
        # return flask_http_response.result.return_response(form)

        result = predict_diabetes(form)

        if int(result) == 1:
            predict = "You probably have diabetes"
        else:
            predict = "You probably doesn't have diabetes"

        # Returns our result to a html page
        return flask.render_template("result.html", predict=predict, stat=result)


if __name__ == "__main__":
    app.run(debug=True)
