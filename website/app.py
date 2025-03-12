from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


app = Flask(__name__)
name = 'there'
state = 'California'

def generate_bar_chart(jobs, similarities):

    num_slices = len(jobs)
    similarities_percentage = [sim * 100 for sim in similarities]
    step_size = max(1, len(px.colors.sequential.Mint_r) // num_slices)
    color = [px.colors.sequential.Mint_r[i * step_size] for i in range(num_slices)]

    fig = go.Figure(data=[go.Bar(
        x=jobs, 
        y=similarities_percentage, 
        marker=dict(color=color),
        text=[f"{round(sim, 1)}%" for sim in similarities_percentage],  # Add the percentage text
        textposition='inside',
    )])

    fig.update_layout(
        title = "Job Match Similarity",
        xaxis_title = "Job Titles",
        yaxis_title= "Match Percentage (%)",
        xaxis=dict(tickangle=45,title_font=dict(family="Optima", size=14, color="#333338")),
        yaxis=dict(title_font=dict(family="Optima", size=14, color="#333338"),tickformat="%", tickmode="array", tickvals=[i for i in range(0, 101, 10)], ticktext=[f"{i}%" for i in range(0, 101, 10)], range=[0, 100]),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin={"t": 30, "b": 30, "l": 50, "r": 50},
    )
    pio.write_image(fig, "static/job_matches_bar_chart.svg")


def generate_radar_chart(job, job_values):
    labels = list(job_values.keys())
    values = list(job_values.values())
    
    df = pd.DataFrame({"Value":labels, "Importance":values})

    fig = px.line_polar(df, r="Importance", theta="Value", line_close=True)

    fig.update_traces(line=dict(color="#0e585f"))

    fig.update_layout(
    title={
        "text": f"<b>Top Attributes for {job}</b>",
        "x": 0.5,
        "xanchor": "center",
        "y": 0.95,
        "yanchor": "top",
        "font": {"family": "Optima", "color": "#333338"}
        },
        legend=dict(
            x=1,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font={"family": "Optima", "color": "#333338"}),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        autosize=True,
        margin={"t": 100, "b": 40, "l": 0, "r": 0},
        width = 1000,
        height = 400
    )

    filename = f"static/{job.replace(' ','_')}.svg"
    pio.write_image(fig, filename)

    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form', methods = ['GET', 'POST'])
def form():
    global name
    name = request.form.get("userName") if request.form.get("userName") else "there"
    return render_template('form.html', placeholder=name)

@app.route('/jobs', methods = ['GET', 'POST'])
def jobs():
    global state
    state = request.form.get("state")
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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X,y)
    user_df = pd.DataFrame(user_input, columns=X.columns)

    distances, indices = knn.kneighbors(user_df)
    jobs = y.iloc[indices[0]].tolist()
        
    user_features = user_input[0].tolist()
    similarities = []
    for i, job in enumerate(jobs):
        job_features = df[df["Title"]==job].iloc[0]
        job_features = job_features.drop(["O*NET-SOC Code", "Title"])
        job_features = pd.to_numeric(job_features)
        job_features = job_features.tolist()
        differences = [abs(user_features[x] - job_features[x]) for x in range(16)]
        total_difference = sum(differences)
        similarity_score = (18 - total_difference) / 18
        similarities.append(similarity_score)
    generate_bar_chart(jobs, similarities)
    
    # Extract top 5 values for radar chart
    df.rename(columns={"Post-Secondary Certificate, Some College, or Associate's Degree":"Some College or Associate's Degree"}, inplace=True)
    for job in jobs:
        job_values = df[df["Title"]==job].iloc[0]
        job_values = job_values.drop(["O*NET-SOC Code", "Title"])
        job_values = pd.to_numeric(job_values)
        top_values = job_values.nlargest(5).to_dict()
        generate_radar_chart(job, top_values)

    return render_template('jobs.html', jobs=jobs)

@app.route('/job/<job_title>')
def job_info(job_title):
    descs = pd.read_csv("descriptions.csv")
    
    match = descs[descs['Title'] == job_title]
    if match.empty:
        description = "Description not available."
    else:
        description = match['Description'].iloc[0]
    
    return render_template('job_info.html', job_title=job_title, description=description, state=state)

if __name__ == '__main__':
    app.run(debug=True)








