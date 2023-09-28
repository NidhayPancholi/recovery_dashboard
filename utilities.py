from deta import Deta  # Import Deta
import joblib
import pandas as pd
# Initialize
deta = Deta("c0eiDit8BAbC_XM839SW9oG1YtfURbZFMfwJVmKQvKYPS")

def read_drive_file(filename):
    drive = deta.Drive("recovery_data")
    file = drive.get(filename).read()
    df=pd.read_excel(file)
    return df

def write_drive_file(filename,model_file):
    drive = deta.Drive("recovery_data")
    drive.put(filename, model_file)
    return True


def save_model(model,filename):
    joblib.dump(model,filename)
    return True

def get_model(filename):
    model=joblib.load(filename)
    return model