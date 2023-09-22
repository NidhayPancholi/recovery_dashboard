from deta import Deta  # Import Deta
import pandas as pd
# Initialize
deta = Deta("c0eiDit8BAbC_XM839SW9oG1YtfURbZFMfwJVmKQvKYPS")

def read_drive_file(filename):
    drive = deta.Drive("recovery_data")
    file = drive.get(filename).read()
    df=pd.read_excel(file)
    return df