import joblib
counter = 0
path  = "config.data"
try:
    counter = joblib.load("config.data")
except:
    counter = 0

def get_counter():
    global counter
    counter += 1
    joblib.dump(counter, path)
    return counter

