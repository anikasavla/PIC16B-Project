### Career Recommendation Service

This project revolves around creating a career recommendation platform for users. We began by working with data from the O*NET SOC database; the various csv files within this repository were collected from the database and further wrangled within the project.ipynb file. We used machine learning to build our recommendation model, and then created a server for the user to be able to interact with our project using Flask. 

In order to work with this project, only the "website" folder is necessary. After dowloading the folder, you can run the following commands in your terminal to launch our website:

```
cd website
python3 app.py
```

It will create a Flask server link (such as http://127.0.0.1:5000) which you can copy and paste into a browser to interact with our project. We recommend running this in a virtual environment, and putting the Flask link into an Incognito tab. 

The following illustrates our process for creating this project as well as our motivations behind it:

The purpose of our project is to assist users in finding career fields that suit their particular experiences, interests, and values. Our target user demographic is college students and young people exploring different career options, or people considering a career switch. One problem that people in this demographic may face is that they know what they value in a career and what they are interested in, but they may have trouble finding careers that align with those interests and that would fit them. Our project bridges that gap by using a database with career data and a machine learning model to identify fitting career paths for the user.

We first identified a fitting data source for our project: the O*NET Resource Center from the U.S. Department of Labor, Employment, & Training. We focused on four key datasets: Occupations; Education, Training, and Experience; Work Values; and Interests. We reformatted the dataset using Pandas dataframes to combine various features, such as combining education levels into a single column and normalizing data for each category to be between 0 and 1. We then built a machine learning model to perform our classification task. We chose a K-Nearest Neighbors classifier to find the data points that were closest/most similar to the user’s inputs. We chose a KNN classifier because it does not require training and instead focuses on the distances between data points, making it easily scalable to very large datasets. We used the 5 nearest neighbors to find the 5 most fitting job recommendations for the user.

After we completed our model, we used Flask to create a website for the user to be able to easily run our model. We first focused on creating a simple website that allowed the user to enter their name and their information such as location, education level, interests, and values, and then receive their 5 most fitting jobs. We then improved the user experience by using Plotly to create visualizations to explain the results to the users. We created a bar chart displayed on the job results page to show the users how close each of the jobs are to their specifications (using a % match). We also created individual pages for each job that can be accessed from the results page and included radar charts on each job’s page to display and visualize the main aspects of each job. We also improved the formatting and aesthetics of the website and improved the accessibility of certain features to improve the user experience.
