# dependencies
import pickle
from flask import Flask
from flask import request
from flask import jsonify

# paths
model_file = "model1.bin"
dv_file = "dv.bin"

# load model
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)
    
# load dict vectorizer
with open(dv_file, "rb") as f_in:
    dv = pickle.load(f_in)
    
# initialize app
app = Flask("churn")

# make route for prediction post request
@app.route("/predict", methods=["POST"])
# define function for prediction
def predict():
    # get data for customer
    customer = request.get_json()
    
    # one hot encode
    X = dv.transform([customer])
    
    # get prediction probalities
    y_pred = model.predict_proba(X)[0, 1]
    
    # get churn decision based on probability
    churn = y_pred >= 0.5
    
    # prepare dict for returning result
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }
    
    # turn result into json and return
    return jsonify(result)

# main method to only run app when explicitly called this script
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
