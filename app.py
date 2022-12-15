from flask import Flask, render_template
import pandas as pd
import numpy as np
import io
import requests
import json
import plotly
from plotly.subplots import make_subplots
from statsmodels.tsa.api import ExponentialSmoothing

app = Flask(__name__)

@app.route("/")
def home():
    # get new data``
    train_dataset = get_new_data()
    # fit model and predict
    # need to specify which country/region, currentlu use the first country
    train = train_dataset.values[:1][0]
    fit = ExponentialSmoothing(train, seasonal_periods=3).fit()
    prediction = fit.forecast(30)
    # polt
    graphJSON = polt_scatter(train, prediction)
    return render_template('notdash.html', graphJSON=graphJSON)


def read_online_csv(url: str, country_or_region: str) -> pd.DataFrame:
    """
    Read the online csv file from given url as DataFrame. Return the Covid-19 data from given country or region without
    the Province/Sate, Lat, and Long.
    """
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode("utf-8"))).drop(["Province/State", "Lat", "Long"], axis = 1)
    data["Country/Region"] = data["Country/Region"].replace("Korea, North", "North Korea")
    data["Country/Region"] = data["Country/Region"].replace("Korea, South", "South Korea")
    data["Country/Region"] = data["Country/Region"].replace("US", "United States")
    return data[data["Country/Region"] == country_or_region].groupby("Country/Region").sum()


def get_new_data(country_or_region: str) -> np.ndarray:
    confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    # deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    # recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    confirmed_data = read_online_csv(confirmed_url, country_or_region)
    # deaths_data = read_online_csv(deaths_url, country_or_region)
    # recovered_data = read_online_csv(recovered_url, country_or_region)

    last_hundred_days_confirmed = confirmed_data.iloc[:, -101:].values[0]
    last_hundred_days_new_confirmed = []
    for i in range(1, len(last_hundred_days_confirmed)):
        new_confirmed = last_hundred_days_confirmed[i] - last_hundred_days_confirmed[i - 1]
        last_hundred_days_new_confirmed.append(new_confirmed)

    return np.array(last_hundred_days_new_confirmed) # , deaths_data, recovered_data


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
