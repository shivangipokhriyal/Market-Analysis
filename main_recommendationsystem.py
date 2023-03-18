from flask import Flask,jsonify, render_template, request, jsonify
import string
import joblib 
import Recommender_System as rs
import numpy as np
import pandas as pd
import jsonpickle



Books = pd.read_csv('Books.csv')
Users = pd.read_csv('Users.csv')
Ratings = pd.read_csv('Ratings.csv')



app = Flask(__name__)
print('Welcome to Deployement')


@app.route('/')
def home():
    return render_template('home_.html')

@app.route('/movie_suggestions_list', methods=['post'])
def movie_recom():
    movie_input=request.form.get('movie_input')
    
    mov_suggest,sim_score= rs.movies_data(Books,Users,Ratings)

    rec = rs.recommendation_system(str(movie_input),mov_suggest,sim_score)
    rec = pd.DataFrame(rec,columns=['Book Name','Author', 'Link'], index=pd.RangeIndex(1,6))

    return render_template("dataframe.html",data=rec.to_html())




app.run(debug=True,port=5000)



