import flask
import jobs
app = flask.Flask(__name__)


@app.route("/")
def app_home():
    return flask.render_template("index.html")


@app.route("/result")
def app_result():
    return flask.render_template("result.html")


app.run(debug=True)
