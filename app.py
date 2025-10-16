import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__,template_folder='mytemp')
model = pickle.load(open("model.pkl", "rb"))

app.route("/")
def Home():
    return render_template(".idk.html")
app.route('/prediction',methods=["POST"])

def predict():

    try:

        float_features = [float(x) for x in request.form.values()]


        features = [np.array(float_features)]


        predictions = model.predict(features)


        predicted_salary = predictions[0]


        return render_template(
            ".idk.html",
            prediction_text=f"The Predicted salary is ${predicted_salary:,.2f}"
        )

    except ValueError:
        return render_template(
            ".idk.html",
            prediction_text="Error: Please ensure all input fields are valid numbers."
        )
    except Exception as e:

        return render_template(
            ".idk.html",
            prediction_text=f"An unexpected error occurred: {e}"
        )
if __name__ == "__main__":
    app.run(debug=True)


