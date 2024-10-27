# dependencies
import pickle

# paths
model_file = "model1.bin"
dv_file = "dv.bin"

# load model
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)
    
# load dict vectorizer
with open(dv_file, "rb") as f_in:
    dv = pickle.load(f_in)
    
# make test customer
customer = {"job": "management", "duration": 400, "poutcome": "success"}

# one hot encode customer
X = dv.transform([customer])

# score the customer
y_pred = model.predict_proba(X)[0, 1]

print(round(y_pred, 3))
