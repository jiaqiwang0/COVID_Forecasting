from flask import Flask, render_template
import pandas as pd
import numpy as np
import json
import plotly
from plotly.subplots import make_subplots
from statsmodels.tsa.api import ExponentialSmoothing
#import requests


app = Flask(__name__)

@app.route("/")
def home():
    # get new data
    train_dataset = get_new_data()
    # fit model and predict
    # need to specify which country/region, currentlu use the first country
    train = train_dataset.values[:1][0]
    fit = ExponentialSmoothing(train, seasonal_periods=3).fit()
    prediction = fit.forecast(30)
    # polt
    graphJSON = polt_scatter(train, prediction)
    return render_template('notdash.html', graphJSON=graphJSON)


def get_new_data():
    """
    TODO get updated data
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    raw_page = requests.get(url)
    print(raw_page.text)
    """
    cofirmed_global_data = pd.read_csv("time_series_covid19_confirmed_global.csv")
    drop_clo = ['Province/State','Country/Region','Lat','Long']
    cofirmed_global_data_modified = cofirmed_global_data.drop(drop_clo,axis=1)
    datewise= list(cofirmed_global_data_modified.columns)
    # recent 100 days data as training data
    train_dataset = cofirmed_global_data_modified[datewise[-100:]]
    #val_dataset = train_dataset[datewise[-30:]]
    return train_dataset


def polt_scatter(train, prediction):
    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(
        plotly.graph_objs.Scatter(x=np.arange(100), mode='lines', y=train, marker=dict(color="dodgerblue"),
                name="Train"), row=1, col=1
    )
    fig.add_trace(
        plotly.graph_objs.Scatter(x=np.arange(100, 130), y=prediction, mode='lines', marker=dict(color="darkorange"),
                name="Val"), row=1, col=1
    )
    fig.update_layout(height=2000, width=1600, title_text="Exponential smoothing")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON