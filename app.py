from flask import Flask, render_template, request
from src.model_predictor import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        user_text = request.form["text"]
        prediction = predict_sentiment(user_text)

        return render_template(
            "result.html",
            text=user_text,
            sentiment=prediction
        )
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)