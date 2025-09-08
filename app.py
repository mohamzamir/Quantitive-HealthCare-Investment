from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
import dash_table

import collections
import bisect

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

data = pd.read_csv('./data/efficient_frontier.csv')
weights = pd.read_csv('./data/weights.csv')
n = len(data)
min_risk = data['Risk'][0]
max_risk = data['Risk'][n - 1]
incr = data['Risk'][0]
# data.insert(-1, "Risk2", np.sqrt(data['Risk']))
risk_return_df = pd.read_csv('./data/predicted_out.csv')
returns = data['Return']
risks = data['Risk']

ind_map = collections.OrderedDict()
ind_map = {
    data['Risk'][i] : int(i) for i in range(n)
}

scatter_df = pd.read_csv('./data/scatter.csv')

trial_info_df = risk_return_df[['NCT', 'Website', 'Condition']]
trial_info_df.insert(2, 'Link', [f"<a href='{website}' target='_blank'>{website}</a>" for website in trial_info_df['Website']])
trial_info_df = trial_info_df.drop(columns=["Website"])

names = risk_return_df['NCT'].tolist()

# Styles
component_box_style = {
    'border': '2px solid #DDD',  # Lighter border for the white box
    'backgroundColor': '#FFFFFF',  # White background for the box
    'padding': '20px',
    'boxSizing': 'border-box',
    'margin': '10px',
    'borderRadius': '5px',  # Rounded corners for the box
}

background_color = '#F0F0F0'  # Slight gray background

header_style = {
    'textAlign': 'center', 
    'color': '#333', 
    'fontWeight': 'bold',  # Make the header text bold
    'margin-bottom' : "0px",
}

app.layout = html.Div([
    html.Div([  # Top Row with corrected vertical separator
        html.Div([  # Top Left Quadrant: Title and Slider
            html.H1("Clinical Trial Investing", style=header_style),
            html.H4("Risk Level", style=header_style),
            dcc.Slider(
                min=min_risk,
                max=max_risk - 0.005,
                step=0.01,
                marks=None,
                value=data['Risk'][n//2],
                id='risk_slider',
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([  # DataFrame display
                dash_table.DataTable(
                    id='table',
                    columns=[{"id": "NCT", "name" : "NCT"},
                             {"id": "Weights", "name" : "Weight (%)"},
                             {"id": "Condition", "name": "Condition"},
                             {"id": "Link", "name" : "Website", "presentation" : "markdown"}],
                    # data=trial_info_df.to_dict('records'),
                    style_cell={
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 10,
                        'textAlign': 'left',
                        'fontFamily': 'Times New Roman'
                    },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'fontFamily': 'Times New Roman'
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'NCT'},
                        'width': '15%'},
                        {'if': {'column_id': 'Weights'},
                        'width': '15%'},
                    ],
                    style_as_list_view=True,
                    page_size=10,
                    markdown_options={"html": True},
                )
            ], style={'marginTop': '20px'}),

        ], style={**component_box_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([  # Top Right Quadrant: Pie Chart
            html.H1("Portfolio Allocation Weights", style=header_style),
            dcc.Graph(id="graph"),
        ], style={**component_box_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'boxSizing': 'border-box', 'alignItems': 'stretch', 'backgroundColor': background_color, 'justifyContent': 'space-between'}),

    html.Div([
        html.H1("Efficient Frontier (Risk Return Trade-off)", style=header_style),
        dcc.Graph(id="plot", style={"margin-top": "0px"}),
    ], style={**component_box_style, 'width': '100%', 'boxSizing': 'border-box'}),
    html.Div([
        html.P("Made by Keigo Hayashi and Jonathan Xu @ Hacklytics 2024", 
               style={
                'text-align': 'center',
                }),
    ], style={**component_box_style, 'width': '100%', 'boxSizing': 'border-box'}),
], style={'padding': '20px', 'boxSizing': 'border-box', 'fontFamily': 'Times New Roman', 'backgroundColor': background_color})



from plotly.subplots import make_subplots
import plotly.graph_objects as go

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input('risk_slider', 'value'))
def generate_chart(risk_slider):
    print("GENERATING GRAPH" + str(risk_slider))
    # ind = bisect.bisect_left(ind_map.keys(), risk_slider)
    risk_slider = round(risk_slider, 3)
    df = pd.DataFrame(zip(names, weights.iloc[ind_map[risk_slider]].values), columns = ['name', 'weight'])
    fig = px.pie(df.sort_values('weight', ascending = [False]), values='weight', names='name', hole=0.3, labels=None)
    fig.update_layout(
        font=dict(family="Times New Roman", size=20, color="RebeccaPurple")  # Update font here
    )
    fig.update_traces(textposition='inside')
    # fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', showlegend=False)
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

# @app.callback(
#     Output('risk-output-container', 'children'),
#     Input('risk_slider', 'value'))
# def update_output(risk_slider):
#     return 'You have selected risk="{}"'.format(risk_slider)

@app.callback(
    Output(component_id='plot', component_property='figure'),
    Input('risk_slider', 'value'))
def update_graph(risk_slider):
    # Create the scatter plot
    fig = px.scatter(scatter_df, x='Risk', y='Return', color='Color')
    fig2 = px.line(data, x='Risk', y='Return')
    fig2.update_traces(line_color='#FFA500', line_width=5)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(showlegend=False)
    fig2.add_scatter(x=[risk_slider], y=[data['Return'][ind_map[risk_slider]]], mode='markers', marker=dict(color='red', size=20), name='Selected Risk Level')
    fig.add_traces(fig2.data)
    fig.update_layout(
        font=dict(family="Times New Roman", size=20, color="RebeccaPurple"),  # Update font here
    )
    return fig

@app.callback(
    Output(component_id='table', component_property='data'),
    Input('risk_slider', 'value'))
def update_graph(risk_slider):
    # Create the scatter plot
    table_data_copy = trial_info_df.copy()
    table_data_copy.insert(0, "Weights", np.round(weights.iloc[ind_map[risk_slider]].values * 100, 2))
    return table_data_copy.sort_values("Weights", ascending=[False]).to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)
# from dash import Dash, dcc, html, Input, Output, callback
# import os


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets)

# server = app.server

# app.layout = html.Div([
#     html.H1('Hello World'),
#     dcc.Dropdown(['LA', 'NYC', 'MTL'],
#         'LA',
#         id='dropdown'
#     ),
#     html.Div(id='display-value')
# ])

# @callback(Output('display-value', 'children'), Input('dropdown', 'value'))
# def display_value(value):
#     return f'You have selected {value}'
