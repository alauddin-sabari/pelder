from dash import Dash, dcc, html, Input, Output
import os











from atexit import register
import dash
import re
# from this import d
from turtle import title
from typing import Tuple
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash_table
from dash.dependencies import  State
from dash.exceptions import PreventUpdate
import pickle
import numpy as np

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets)

app =dash.Dash(__name__,
title="Patrick",
                 assets_folder ="static",
                 assets_url_path="static",
          external_stylesheets=[dbc.themes.SLATE])#[dbc.themes.BOOTSTRAP])
server = app.server


app.config.suppress_callback_exceptions=True         

app.title = 'Patrick Elder'

df = pd.read_csv('CA_low_to_high.csv')
#######################-----------------------Scatter Map+other-------------------------###########################

single_price_pooint_slider = dcc.Slider(0,80, value=0, step=0.125,
        id = 'slider-input',
        marks={ 1: '1',  1.25:'1.25 ' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},

           included=False,  # 
    tooltip={"placement": "bottom", "always_visible": True}
           )
#___________________________________Slider__________________________________
range_slider = dcc.RangeSlider(           # Acre Size      
        id='range-slider',  
        min=0, max=80, step=0.125,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        marks={ 1: '1',  1.25:'1.25 ' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        
        #marks={i: '{}'.format(2 ** i) for i in range(7)},
        value=[0.125, 1],
        tooltip={"placement": "bottom", "always_visible": True}
        
    )
percent_slider = dcc.Slider(           # Acre Size      
        id='percent-slider',  
        min=.2, max=1,#value=0.25,
         step=0.01,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        marks={ .20: '20%', .25: '25%',  .30:'30%' ,.35:'35%', .40:'40%', .50: '50%', .60: '60%',  .70:'70%' ,.80:'80%', .90:'90%', 1: '100%'},
        
        #marks={i: '{}'.format(2 ** i) for i in range(7)},
        # value=[.25, .30],
        
        tooltip={"placement": "bottom", "always_visible": True}
        
    )
price_slider = dcc.RangeSlider(           # Acre Size      
        id='price-slider',  
        min=1000, max=250000, step=1000,
        # marks={0: '0', 2.5: '2.5'},
        
        marks={ 1000: '1k',  5000:'5k ' , 10000:'10k',20000:'20k', 50000:'50k', 100000:'100k', 250000:'250k'},
        
        # marks={i: '{}'.format(100 ** i) for i in range()},
        value=[1000, 250000],
        tooltip={"placement": "bottom", "always_visible": True}
        
    )
#___________________________________Map_______________________________________

map_graph = dbc.Container(
    
    dcc.Graph(id= "map-plot", config={#isplaylogo': False}) # config -> https://swdevnotes.com/python/2020/plotly_customise_toolbar/
                 "displaylogo": False,
                 'modeBarButtonsToRemove': [
                    #  'zoom2d',
                    #  'toggleSpikelines',
                    #  'pan2d',
                    'drawcircle',
                     'select2d',
                    #  'lasso2d',
                    'eraseshape'
                    
                     'autoScale2d',
                     'hoverClosestCartesian',
                     'hoverCompareCartesian'],

                     'modeBarButtonsToAdd':['toggleSpikelines',
                                            # 'drawline',
                     
                                        # 'drawopenpath',
                                        # 'drawclosedpath',
                                        # 'drawcircle',
                                        # 'drawrect',
                                        # 'eraseshape'
                                       ]
                     })
)

#___________________________________Drop-Down_______________________________________
#https://ontheworldmap.com/usa/state/california/map-of-california-and-nevada.html states and cities

drop_down = html.Div([
                    dcc.Dropdown(id='data-set-chosen', multi=False, value='California',
                     options=[{'label':'California', 'value':'California'},
                              {'label':'Nevada', 'value':'California'},
                              {'label':'Arizona', 'value':'Arizona'}])
    ], className='row', style={'width':'50%'}),                                                 #dcc.Dropdown(['California', 'Nevada', 'SF'], 'NYC', id='data-set-chosen')
#___________________________________Box_Plot___________________________________

Box_plot = dbc.Container(
    
    dcc.Graph(id= "box-plot",config= {'displaylogo': False})
)
#___________________________________Reg_Plot______________________________________ 

reg_plot =dbc.Container(
    
    dcc.Graph(id= "reg-plots",config= {'displaylogo': False})
)

#___________________________________Table_______________________________________

            # #______________________Polygon________________

filter_table = html.Div([
    # html.Button("Download Excel", id="btn_xlsx"),

    dbc.Button(id = 'btn_xlsx',
            children=[html.I(className="fa fa-download mr-1"), "Save  filtered data by slider"],
            color="info",
            className="mt-1"),
            #____________________________________________________________showing data table below_______________
 #_New__________   
    html.Div([
        html.Div(id='table-placeholder', children=[])
    ], className='row'),
# #______________________Polygon________________
dbc.Button(id = 'btn_xlsx-polygon',
            children=[html.I(className="fa fa-download mr-1"), "Save Polygon data"],
            color="info",
            className="mt-1"),
 html.Div([
        html.Div(id='table-placeholder-polygon', children=[])
    ], className='row'),
#     dash_table.DataTable( id ='polygon-table',

#                         columns = [{'name':i, 'id':i}  for i in df_table.columns],
#                         fixed_rows = {'headers': True, 'data': 1},
#                         data = df_table.to_dict('records'),
#     style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
#     style_table={'height': '300px', 'overflowY': 'auto'},
#                     ),
            #___________________________________________________________ending table show_______

    
    dcc.Download(id="download-dataframe-xlsx"),
    dcc.Download(id="download-dataframe-xlsx-polygon"),

])


marginal_distribution_plot =dbc.Container(
    
    dcc.Graph(id= "marginal-distribution-plots",config= {'displaylogo': False})
)

#######################-----------------------End-------------------------###########################
def drawText(id, title_):
    return html.Div([
        dbc.Card(
             
            dbc.CardBody([
                html.H4(title_, className='card-title fw-bold' ),
                html.Div([
                    html.H2(id=id, className='text-warning'),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])
crd = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText('avg', 'Average Price')
                ], width=2),
                dbc.Col([
                    drawText('avg_4','Average/4 Price')
                ], width=2),
                dbc.Col([
                    drawText('pred','Predicted Price')
                ], width=2),
                dbc.Col([
                    drawText('q1', 'Quartile 1 Price:')
                ], width=3),
                dbc.Col([
                    drawText('q2', 'Quartile 2 Price:')
                ], width=3),
            ], align='center'), 
                 
        ]), color = 'dark'
    )
])
#######################-----------------------App Layout-------------------------###########################

app.layout = html.Div([
    html.H1(html.Mark("Patrick's Property listing App"), className= 'opacity-100 bg-secondary p-3 m-2 text-info text-center  fw-bold rounded'),
 
 dbc.Row([single_price_pooint_slider ,
                 html.Div(id='slider-output') ]),
    crd,  #card all tile above
    
    dbc.Row([
        dbc.Col( [

                          # must double bracket
                dbc.Row(range_slider),  # should be 
                
                dbc.Row(map_graph),  # should be 
                dbc.Row(price_slider),
    ], width= 8),

      
    
        # dbc.Col(None, width= 2),

        # dbc.Col(Box_plot, width=4)
        dbc.Col([dbc.Row(percent_slider),
        html.Br(),
                dbc.Row(drop_down),  # should be 
                html.Br(),
                dbc.Row(reg_plot)]
                , width=4)
        

    ]),
    dbc.Row([

      
         # Clustering  bottom right side
        dbc.Col(marginal_distribution_plot, width=5),
        dbc.Col(Box_plot, width=2),

        dbc.Col( filter_table, width= 5)

        

    ]),

     # step- 1
    #__________________________Store Data in here______________________________

    dcc.Store( id = 'store-slider-data', data = [], storage_type = 'memory'),
    dcc.Store( id = 'store-polygon-data', data = [], storage_type = 'memory'),
    dcc.Store( id = 'store-dropdown-data', data = [], storage_type = 'memory'),


     
     


])





#######################-----------------------Main-Function-------------------------###########################
#______________________________________________topbar slider single point
# @app.callback(
#     Output('slider-output', 'children'),
#     Input('slider-input',  'value')
# ) 
# def output_from_slider(value):
#     price = value*100
#     return  f"Your acre size {value} vs  price: {price}"

# def top_bar_price(df1):
#         # print('\n------------\n-------len df',len(df1))
#         ln_df1 = len(df1)
#         if ln_df1 ==0:# or number_of_points==0:
#             return [0,0,0,0]
#         else:
#             avg = int(df1['Price'].mean())
#             avg_f= f'${avg}'  #since later step it will be devide then get error

#             avg_by_4 =int(avg/4) 
#             avg_by_4 = f'${avg_by_4}'
#             Q1 = f'${int(df1.Price.quantile(0.25))}'
            
#             print(Q1)
#             Q3 =f'${int(df1.Price.quantile(0.75))}' 
#             ls = [avg_f, avg_by_4, Q1, Q3]
#             return ls
@app.callback(
    # Output('avg', 'children'),
    # Output('avg_4', 'children'),
    # Output('q1', 'children'),
    # Output('q2', 'children'),
    Output('slider-output', 'children'),
                                      
    # Input("range-slider", "value"),
    Input("slider-input", "value"),

    Input("price-slider", "value"),
                                                                    
    Input(component_id="map-plot", component_property='selectedData')     # Extra input for drawing polygon.
)
                            
def top_single_slider_acre_vs_price(acr_range,price_range, slct_data):  # this acre is a single value not touple.
    global df
    # df = pd.read_csv('CA_combined.csv')
    if acr_range==0:
        low_range = 0
        high_range = 0

    elif acr_range <=0.125:
        low_range = acr_range -0.067
        high_range = acr_range + 0.067

    elif 0.125<acr_range <=0.25:

        low_range = acr_range -0.1 
        high_range = acr_range +0.1

    elif 0.25<acr_range <=0.5:

        low_range = acr_range -0.125
        high_range = acr_range +0.125

    elif 0.5<acr_range <=1:


        low_range = acr_range -0.15
        high_range = acr_range +.15
    elif 1<acr_range <=1.25:


        low_range = acr_range -0.25
        high_range = acr_range +0.25

    elif 1.25< acr_range <=2.5:


        low_range = acr_range - 0.35
        high_range = acr_range + 0.35

    elif 2.5< acr_range <5:


        low_range = acr_range -1
        high_range = acr_range +1

    elif 5< acr_range <=10:  
        low_range = acr_range -1.25
        high_range = acr_range +1.25

    elif 10< acr_range <15:  
        low_range = acr_range -1.5
        high_range = acr_range +1.5

    elif 15< acr_range <20:  
        low_range = acr_range - 1.75
        high_range = acr_range + 1.75 

    elif 20< acr_range <=20:  
        low_range = acr_range - 2
        high_range = acr_range + 2   

    elif 30< acr_range <=40:  
        low_range = acr_range - 3
        high_range = acr_range + 3       
    else:
        low_range = acr_range - 4
        high_range = acr_range +4     

    acr_range = [low_range, high_range]
    #df1 = data_filter_by__slider(df, acr_range, price_range)
    df1 = pd.read_csv('CA_low_to_high.csv')    # we will use df when we don't wanna connect function with range slider.

    print('\n----------df inside average callback functions\n', df1)
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
        r = top_bar_price(df1)
        return r[1]
        
        # return f'Average Price : ${int(avg)} \n  ||   targeted price (avg/4) : ${int(avg/4)}  [# Q1: ${Q1}   # Q3: ${Q3}]'

    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        r = top_bar_price(df1)
        return r[1]
#___________________________________End_______________________________________








# df = pd.read_csv('drawing_polygon_without_outliers1.csv')    # we will use df when we don't wanna connect function with range slider.
# df = pd.read_csv('CA_combined1.csv')
df = pd.read_csv('CA_low_to_high.csv')
#___________________________________Table_______________________________________

def data_filter_by__slider(df, acr_range, price_range):     # always need to call first this function for each callback to get filter data by slider range.
     
   
    low, high = acr_range
    
    mask = (df['Lot_Size'] > low) & (df['Lot_Size'] < high)
    df1 = df.loc[mask]                                                   # never reassign   df = df[ something]  df1 or something else
    # if price_range:
    low_price, high_price = price_range
                                                                        # make sure to use df1 to right side
    mask = (df1['Price'] > low_price) & (df1['Price'] < high_price)
    # df2 = df1.l[mask] 
    df2 = df1.loc[mask] 

                                                
    return df2
#___________________________________Map-plot_______________________________________


def map_plt(df):
    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(
        df,  lat='latitude', lon='longitude',     color="Price", size="Lot_Size",
                #   color_continuous_scale=px.colors.cyclical.IceFire,    
        title = "US    Property Listing",

        # hover_data=1,
                        hover_name='Address',
#                         text='Address',
        
        zoom=5, mapbox_style='open-street-map')
    fig.update_layout(
        title = "US    Property Listing",
    autosize = True,
    # width = 1350,
    height = 750
)
    # fig.update_layout(
    #     mapbox_style="white-bg",
    #     mapbox_layers=[
    #         {
    #             "below": 'traces',
    #             "sourcetype": "raster",
    #             "sourceattribution": "United States Geological Survey",
    #             "source": [
    #                 "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
    #             ]
    #         }
    #     ])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}),
    
    fig.update_layout(mapbox_style="open-street-map") 

    return fig


@app.callback(
    Output("map-plot", "figure"), 
    Input("range-slider", "value"),
    Input("price-slider", "value")
    )

    

def update_map_chart(acr_range,price_range):
    global df
    df1 = data_filter_by__slider(df, acr_range, price_range) 
    m = map_plt(df1)
    return m
        

#__________________________________Top _______________________________________


def top_bar_price(df1):
        # print('\n------------\n-------len df',len(df1))
        ln_df1 = len(df1)
        if ln_df1 ==0:# or number_of_points==0:
            return [0,0,0,0]
        else:
            avg = int(df1['Price'].mean())
            avg_f= f'${avg}'  #since later step it will be devide then get error

            avg_by_4 =int(avg/4) 
            avg_by_4 = f'${avg_by_4}'
            Q1 = f'${int(df1.Price.quantile(0.25))}'
            
            print(Q1)
            Q3 =f'${int(df1.Price.quantile(0.75))}' 
            ls = [avg_f, avg_by_4, Q1, Q3]
            return ls
@app.callback(
    Output('avg', 'children'),
    Output('avg_4', 'children'),
    Output('q1', 'children'),
    Output('q2', 'children'),
                                      
    Input("range-slider", "value"),
    Input("price-slider", "value"),
    # Input("percent-slider", "value"),

                                                                    
    Input(component_id="map-plot", component_property='selectedData')     # Extra input for drawing polygon.
)
                            
def average_finder(acr_range,price_range, slct_data):
    global df
    

    df1 = data_filter_by__slider(df, acr_range, price_range)
    # df1 = pd.read_csv('drawing_polygon_without_outliers1.csv')    # we will use df when we don't wanna connect function with range slider.

    print('\n----acre------df inside average callback functions\n', acr_range)
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
        r = top_bar_price(df1)
        return r
        
        # return f'Average Price : ${int(avg)} \n  ||   targeted price (avg/4) : ${int(avg/4)}  [# Q1: ${Q1}   # Q3: ${Q3}]'

    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        r = top_bar_price(df1)
        return r

#___________________________________Box-plot_______________________________________


def box_plt(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Box(

            y = df['Price'],

            name= 'Price',
            marker_color='Orange',
            boxmean=True,

        )
    )
    fig.update_layout(
        # title = "Property IQR",     this also valid
        title  = {
            'text': "Property's IQR",
            'y':0.9,
            'x':0.5,
            'xanchor':'center',
            'yanchor': 'top'
        },

        autosize=True,
        # width=1800,
        height=750,)
    return fig
@app.callback(
    Output('box-plot', 'figure'),
    Input("range-slider", 'value'),
    Input("price-slider", "value")

    # Input(component_id="scatter-plot", component_property='selectedData')
    
)
    
def set_display_children(acr_range, price_range):
    global df


    df1 = data_filter_by__slider(df, acr_range, price_range)
    
    b = box_plt(df1)
    return b


#___________________________________Reg-plot_______________________________________

def regress_plot(df):

    import plotly.graph_objects as go
    df = pd.read_csv('CA_low_to_high.csv') 
    df = df[df['Price']<250000]
    Q1 = df.Price.quantile(0.25)
    Q3 = df.Price.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    lower_limit, upper_limit
    df_no_outlier = df[(df.Price>lower_limit)&(df.Price<upper_limit)]



    df = df_no_outlier.copy()
    df['Lot_Size'].max()

    df['Unit'].unique()

    df['Lot_Size'][df['Unit']=='sqft lot'] =df['Lot_Size'][df['Unit']=='sqft lot'] .map(lambda x: x/43560)


    df  = df[df["Lot_Size"]<=80]


    X =  df[['Lot_Size']].values
    # X_train, X_test, y_train, y_test = train_test_split(X, df.Price/4, random_state=0)

    # model = LinearRegression()

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    
    y = df.Price/4
    with open('Ca_ml_model_pickle', 'rb') as file:
        mp = pickle.load(file)
    x_range = np.linspace(X.min(), X.max(), 100)
    # y_range = model.predict(x_range.reshape(-1, 1))
    y_range = mp.predict(x_range.reshape(-1, 1))



    fig = go.Figure([
        # go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X.squeeze(), y=y, name='Data point', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='Price')
    ])
    fig.update_layout(
        autosize=False,
        width=1200,
        height=900,)



    fig.update_layout({'plot_bgcolor': "#21201f", 'paper_bgcolor': "#21201f", 'legend_orientation': "h"},
                    legend=dict(y=1, x=0),
                    font=dict(color='#dedddc'), dragmode='pan', hovermode='x unified',
                    margin=dict(b=20, t=0, l=0, r=40))

    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,
                    showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid')

    fig.update_xaxes(showgrid=True, zeroline=True, rangeslider_visible=True, showticklabels=True,
                    showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid')

    fig.update_layout(hoverdistance=0)

    fig.update_traces(xaxis='x', hoverinfo='none')
    fig.update_layout(yaxis_range = [0, 300000])
    return fig

@app.callback(
    Output('reg-plots', 'figure'),
    Input("range-slider", 'value'),
    Input("price-slider", "value")
    )
    
def reggs(acr_range, price_range):
    global df
    

    df1 = data_filter_by__slider(df, acr_range, price_range)
    len_df = len(df1)

    if len_df > 0:
        df1 =df1
    else:
        df1 = df
    b = regress_plot(df1)
    return b



#___________________________________Table for download_______________________________________

#___________________________________Table for download_______________________________________

def data_filter_by__slider_and_storing(df, acr_range, price_range):     # always need to call first this function for each callback to get filter data by slider range.
     
   
    low, high = acr_range
    
    mask = (df['Lot_Size'] > low) & (df['Lot_Size'] < high)
    df1 = df.loc[mask]                                                   # never reassign   df = df[ something]  df1 or something else
    # if price_range:
    low_price, high_price = price_range
                                                                        # make sure to use df1 to right side
    mask = (df1['Price'] > low_price) & (df1['Price'] < high_price)
    # df2 = df1.l[mask] 
    df2 = df1.loc[mask] 

                                                
    return df2
                #___________________________________storing table  by slider_______________________________________

@app.callback(
    Output('store-slider-data', 'data'),
    Input("range-slider", "value"),
    Input("price-slider", "value")
    # Input('data-set-chosen', 'value')
)
def store_the_data(acr_range,price_range):
    global df
    global dataset

  
    dataset = data_filter_by__slider(df, acr_range, price_range) 
                                                  

    return dataset.to_dict('records')


                #___________________________________storing filter table  by  Polygon_______________________________________

@app.callback(
    Output('store-polygon-data', 'data'),
    # Input("range-slider", "value"),
    # Input("price-slider", "value"),
    Input(component_id="map-plot", component_property='selectedData')     # Extra input for drawing polygon.

)
def store_the_data(slct_data):
    global df
    # global dataset
    dataset = df.copy()

   
    
    #df = dataset.copy()                                                # print('\n___________________inside store_the_data and dataset Shape\n',dataset.shape)
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
         
        dataset= df
        
        
    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        dataset = df1
    return dataset.to_dict('records')










#___________________________________Showing slider filtered table______________________________________


@app.callback(
    Output('table-placeholder', 'children'),
    Input('store-slider-data', 'data'),
    # Input("btn_xlsx", "n_clicks"),

)
def create_graph1(data):
    dff = pd.DataFrame(data)
    # dff =df_d [['Address', 'Lot_Size', 'Price']]
    my_table = dash_table.DataTable( id='tbl',
        columns=[{"name": i, "id": i} for i in dff.columns],
        fixed_rows = {'headers': True, 'data': 1},
        data=dff.to_dict('records'),
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
    return my_table

                                    #________polygon_________

@app.callback(
    Output('table-placeholder-polygon', 'children'),
    Input('store-polygon-data', 'data'),
    # Input("btn_xlsx", "n_clicks"),

)
def create_graph1(data):
    dff = pd.DataFrame(data)
    # dff =df_d [['Address', 'Lot_Size', 'Price']]
    my_table = dash_table.DataTable( id='tbl-polygon',
        columns=[{"name": i, "id": i} for i in dff.columns],
        fixed_rows = {'headers': True, 'data': 1},
        data=dff.to_dict('records'),
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
    return my_table

#___________________________________Download-table_______________________________________
import io

@app.callback(
    Output("download-dataframe-xlsx", "data"),
    # Input('store-our-data', 'data'),
    
    Input("btn_xlsx", "n_clicks"),
    # Input('table-placeholder', 'data'),
    State ('tbl', 'data'),
    prevent_initial_call=True,
    
)
def func( n_clicks, table_data):
    
    # df = data
    
    df = pd.DataFrame.from_dict(table_data)
    # df = df[['Address', 'Lot_size', 'Price']]
    if not n_clicks:
        raise PreventUpdate
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=True)
    download_buffer.seek(0)
    return dict(content=download_buffer.getvalue(), filename="patrick.csv")





@app.callback(
    Output("download-dataframe-xlsx-polygon", "data"),
    # Input('store-our-data', 'data'),
    
    Input("btn_xlsx-polygon", "n_clicks"),
    # Input('table-placeholder', 'data'),
    State ('tbl-polygon', 'data'),
    prevent_initial_call=True,
    
)
def func( n_clicks, table_data):
    
    # df = data
    
    df = pd.DataFrame.from_dict(table_data)
    # df = df[['Address', 'Lot_size', 'Price']]
    if not n_clicks:
        raise PreventUpdate
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=True)
    download_buffer.seek(0)
    return dict(content=download_buffer.getvalue(), filename="patrick.csv")






#__________________________________marginal_distribution_plot_______________________________________
#__________________________________________________________________________
#


def marginal_distribution_plot(df):
    
    
    fig = px.density_heatmap(df, x="Lot_Size", y="Price", marginal_x="violin", marginal_y="violin")

    fig.update_layout(
    title  = {
                'text': "Marginal Distribution Plot ",
                'y':0.9,
                'x':0.5,
                'xanchor':'center',
                'yanchor': 'top'
            },
        xaxis = dict(title = "Lot Size"),
        yaxis = dict(title = " Lot count for each Price Range" ),
                    
                    
        autosize=True,
        # width=1200,
        height=725,)
    return fig

@app.callback(
    Output('marginal-distribution-plots', 'figure'),
    Input("range-slider", 'value'),
    Input("price-slider", "value")
    )
    
def reggs(acr_range, price_range):
    global df
    

    df1 = data_filter_by__slider(df, acr_range, price_range)
    len_df = len(df1)

    if len_df > 0:
        df1 =df1
    else:
        df1 = df
    b = marginal_distribution_plot(df1)
    return b



















if __name__ == '__main__':
    app.run_server(debug=True, port= 8059)
