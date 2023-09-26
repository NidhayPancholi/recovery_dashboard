import pickle
model_filename = 'model.pkl'
# Load the saved model
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)


def prediction(row):
    # Make prediction
    prediction = loaded_model.predict(row)
    return prediction

def absolute_error(row):
    # Make prediction
    prediction = loaded_model.predict(row)
    return abs(prediction - row['recoveries'])

# Now 'loaded_model' is the loaded scikit-learn model
