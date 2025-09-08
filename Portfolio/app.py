from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output

import collections
import bisect

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('./efficient_frontier.csv')
weights = pd.read_csv('./weights.csv')
n = len(data)
min_risk = data['Risk'][0]
max_risk = data['Risk'][n - 1]
incr = data['Risk'][0]

names = pd.read_csv('./predicted.csv')['Name'].tolist()
# print(names)
# print(weights.iloc[0].values)

ind_map = collections.OrderedDict()
ind_map = {
    data['Risk'][i] : int(i) for i in range(n)
}

print(ind_map)

# print(ind_map[data['Risk'][9]])
app.layout = html.Div([
    dcc.Slider(min_risk, max_risk - 0.005, 0.001,
               marks = None,
               value = data['Risk'][n//2],
               id='risk_slider',
               tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Graph(id="graph"),
    dcc.Graph(id="plot"),
    html.Div(id='risk-output-container')
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input('risk_slider', 'value'))
def generate_chart(risk_slider):
    print("GENERATING GRAOH" + str(risk_slider))
    # ind = bisect.bisect_left(ind_map.keys(), risk_slider)
    risk_slider = round(risk_slider, 3)
    df = pd.DataFrame(zip(names, weights.iloc[ind_map[risk_slider]].values), columns = ['name', 'weight'])
    fig = px.pie(df, values='weight', names='name', hole=0.3)
    return fig

@app.callback(
    Output('risk-output-container', 'children'),
    Input('risk_slider', 'value'))
def update_output(risk_slider):
    return 'You have selected risk="{}"'.format(risk_slider)

@app.callback(
    Output(component_id='plot', component_property='figure'),
    Input('risk_slider', 'value'))
def update_graph(risk_slider):
    # Create the scatter plot
    fig = px.scatter(data, x='Risk', y='Return', title='Scatter plot of Data Points')
    fig.add_scatter(x=[risk_slider], y=[data['Return'][ind_map[risk_slider]]], mode='markers', marker=dict(color='red', size=10), name='User Point')
    return fig

if __name__ == '__main__':
    app.run(debug=True)