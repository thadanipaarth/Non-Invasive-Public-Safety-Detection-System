import flask
import pandas as pd
import pickle

with open(f'./Models/Safety-Detection-Engine/Model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def home():
    if flask.request.method == 'GET':
        return(flask.render_template('home.html'))
    if flask.request.method == 'POST':
        Mask=flask.request.form['Mask']
        Fever= flask.request.form['Fever']
        Tiredness = flask.request.form['Tiredness']
        DryCough = flask.request.form['Dry-Cough']
        Difficulty= flask.request.form['Difficulty-in-Breathing']
        SoreThroat = flask.request.form['Sore-Throat']
        NasalCongestion = flask.request.form['Nasal-Congestion']
        RunnyNose = flask.request.form['Runny-Nose']
        Gender_Female = flask.request.form['Gender_Female']
        Gender_Male = flask.request.form['Gender_Male']
        input_variables = pd.DataFrame([[Fever, Tiredness,DryCough, Difficulty,SoreThroat,
                                         NasalCongestion,RunnyNose,Gender_Female,Gender_Male]],
                                       columns=['Fever','Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing','Sore-Throat','Nasal-Congestion','Runny-Nose','Gender_Female','Gender_Male'],
                                       dtype=float)
        prediction= model.predict(input_variables)[0]
        return flask.render_template('home.html',
                                     original_input={'Fever':Fever,
                                                     'Tiredness':Tiredness,
                                                     'Dry-Cough':DryCough,
                                                     'Difficulty-in-Breathing':Difficulty,
                                                     'Sore-Throat':SoreThroat,
                                                     'Nasal-Congestion':NasalCongestion,
                                                     'Runny-Nose':RunnyNose,
                                                     'Gender_Female':Gender_Female,
                                                     'Gender_Male':Gender_Male
                                                     },
                                     result_health=prediction,
                                     result_mask= Mask
                                     )