from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


app = Flask(__name__)

def generate_pie_chart(jobs, percent_matches):

    num_slices = len(jobs)
    step_size = max(1, len(px.colors.sequential.Agsunset) // num_slices)
    colors = [px.colors.sequential.Agsunset[i * step_size] for i in range(num_slices)]

    fig = go.Figure(data=[go.Pie(labels=jobs, values=percent_matches, pull=[0,0,0.2,0],marker=dict(colors=colors))])
    fig.update_layout(title={"text": "<b>Job Match Percentages</b>",
                             "x":0.5,
                             "xanchor":"center",
                             "y":0.85,
                             "yanchor":"top"},
                      legend=dict(x=1,y=0.5,xanchor="left",yanchor="middle"))

    pio.write_image(fig, "static/job_matches.svg")

    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form', methods = ['GET', 'POST'])
def form():
    userName = request.form.get("userName", "").strip()
    if not userName: 
        userName = "there"
    return render_template('form.html', placeholder=userName)

@app.route('/jobs', methods = ['GET', 'POST'])
def jobs():
    education_level = request.form.get("education")
    user_input = np.array([[
    1 if education_level == "highschool" else 0,
    1 if education_level == "associates" else 0,
    1 if education_level == "bachelors" else 0,
    1 if education_level == "postgrad" else 0,
    float(request.form.get("achievement", 0)),
    float(request.form.get("independence", 0)),
    float(request.form.get("recognition", 0)),
    float(request.form.get("relationships", 0)),
    float(request.form.get("support", 0)),
    float(request.form.get("working_conditions", 0)),
    float(request.form.get("artistic_aspect", 0)),
    float(request.form.get("conventional_aspect", 0)),
    float(request.form.get("enterprising_aspect", 0)),
    float(request.form.get("investigative_aspect", 0)),
    float(request.form.get("realistic_aspect", 0)),
    float(request.form.get("social_aspect", 0))
    ]])

    df = pd.read_csv("jobs.csv")
    X = df.drop(columns=["O*NET-SOC Code", "Title"])
    y = df["Title"]
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X,y)
    user_df = pd.DataFrame(user_input, columns=X.columns)
    jobs = y.iloc[knn.kneighbors(user_df)[1][0]].tolist()[:5]

    # Convert distances to percentage match scores for data visualization
    distances, indices = knn.kneighbors(user_df)
    max_distance = max(distances[0]) if max(distances[0]) > 0 else 1
    percent_matches = [(1-(d/max_distance))*100 for d in distances[0]]
    generate_pie_chart(jobs, percent_matches)

    df2 = pd.read_csv("descriptions.csv")
    descs = []
    for job in jobs:
        match = df2[df2['Title'] == job]
        if match.empty:
            descs.append("Unavailable")
        else:
            descs.append(match['Description'].iloc[0])
    return render_template('jobs.html', jobs=zip(jobs, descs))

if __name__ == '__main__':
    app.run(debug=True)








