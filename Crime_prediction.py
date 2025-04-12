import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from predict_crime import predict
from shapely.geometry import  Point
import matplotlib.pyplot as plt
from matplotlib import cm
import geopandas as gpd
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

st.title("Location Based Crime Prediction")
st.image("https://hhsadvocate.com/wp-content/uploads/2021/12/TrueCrime-2.jpeg")
st.markdown("***")


a,b,c = st.columns(3)
year = c.number_input(label="Enter Year",value=2022,min_value=1900,max_value=3000)

date_in = a.number_input("Enter Date",value=29,max_value=31,min_value=1)

month_in = b.number_input("Enter Month",value=7,max_value=12,min_value=1)

min_in = b.number_input(label="Minutes",value=15,max_value=59,min_value=0)

hour_in = a.number_input(label="Hour",value=23,max_value=23,min_value=0)

sec_in = c.number_input(label="Seconds",value=0,max_value=59,min_value=0)

c,d = st.columns(2)
latitude = c.number_input(label="Latitude",value=22.720992,format="%f")
longitude = d.number_input(label="Longitude",value=75.876083,format="%f")

enter = st.button(label="Predict Crime")

st.markdown("***")
if enter:
    if len(str(month_in)) == 1:
        month_in = "0"+str(month_in)
    if len(str(hour_in)) == 1:
        hour_in = "0"+str(hour_in)
    if len(str(sec_in)) == 1:
        sec_in = "0"+str(sec_in)

    main_date = '-'.join([str(year),str(month_in),str(date_in)])
    main_time = ':'.join([str(hour_in),str(min_in),str(sec_in)])

    date_time = " ".join([main_date,main_time])
    
    sr = pd.Series([date_time])
    sr = pd.to_datetime(sr)
    print(sr)
    print(type(sr.dt.month))

    main_ins = {
              "month": sr.dt.month.tolist()[0],
              "day": sr.dt.day.tolist()[0],
              "hour": sr.dt.hour.tolist()[0],
              "dayofyear": sr.dt.dayofyear.tolist()[0],
              "week": sr.dt.week.tolist()[0],
              "weekofyear": sr.dt.weekofyear.tolist()[0],
              "dayofweek": sr.dt.dayofweek.tolist()[0],
              "weekday": sr.dt.weekday.tolist()[0],
              "quarter": sr.dt.quarter.tolist()[0],
             }
    inputs = list(main_ins.values())
    inputs.append(latitude)
    inputs.append(longitude)

    inputs_arr = np.array(inputs)

    labels = ['Robbery','Gambling','Accident','Violence','Murder','Kidnapping']
    vals,dicts = predict(inputs_arr)
    # print(list(vals.values()))
    explode = (0.2, 0.2, 0.2, 0.2,0.2,0.2)
    plt.pie(vals,labels=labels,explode=explode,shadow=True,autopct='%1.2f%%')
    plt.savefig('images/predict_graph.png')

    st.markdown("> Probablity of types of crimes happening on {} at {}".format(main_date,main_time))
    st.image('images/predict_graph.png')
    st.markdown("> Acts of Crime")
    st.write(dicts)
    
st.markdown("***")
st.markdown("Evaluating the position of the data points using the coordinates")
df = pd.read_csv("main_data.csv")

def create_gdf(df):
    gdf = df.copy()
    gdf['Coordinates'] = list(zip(gdf.latitude, gdf.longitude))
    gdf.Coordinates = gdf.Coordinates.apply(Point)
    gdf = gpd.GeoDataFrame(
        gdf, geometry='Coordinates', crs={'init': 'epsg:4326'})
    return gdf

geo_df = create_gdf(df)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(color='white', edgecolor='black')
geo_df.plot(ax=ax, color='red')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


## gettting the onehotencoded column of crime type category back in single column for better visulaization.  

df_sub = df[['act379', 'act13', 'act279','act323','act363', 'act302']]
df_sub['type'] = df_sub.idxmax(1)
type_column = df_sub.type
df = df.join(type_column)

st.markdown("***")
st.markdown("Number of crimes happened(realtive)")

plt.figure(figsize = [4,4])
df.type.value_counts(normalize = True).plot.barh()
st.pyplot()

st.markdown("***")
st.markdown("Year and crime category relationship")
df.groupby(['year', 'type']).size().plot(kind='barh', figsize=(20, 20), grid=True)
st.pyplot()


st.markdown("***")
st.markdown("hour of the day, crime type and year relationship")
a = df.pivot_table(index='hour', columns='type', values=['year'], aggfunc='size')
a.plot(kind='bar', figsize=(12, 5))
st.pyplot()

st.markdown("***")
st.markdown("Getting the insights into crime happening per week day")
data = df.groupby('weekday').count().iloc[:, 0]
data = data.reindex([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0
])

plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        data.index, (data.values / data.values.sum()) * 100,
        orient='v',
        palette=cm.ScalarMappable(cmap='Reds').to_rgba(data.values))

plt.title('Incidents per Weekday', fontdict={'fontsize': 16})
plt.xlabel('Weekday')
plt.ylabel('Incidents (%)')
st.pyplot()


st.markdown("***")
st.markdown("distribution of number of incidents per day")
col = sns.color_palette()

plt.figure(figsize=(10, 6))
data = df.groupby('day').count().iloc[:, 0]
sns.kdeplot(data=data, shade=True)
plt.axvline(x=data.median(), ymax=0.95, linestyle='--', color=col[1])
plt.annotate(
    'Median: ' + str(data.median()),
    xy=(data.median(), 0.004),
    xytext=(200, 0.005),
    arrowprops=dict(arrowstyle='->', color=col[1], shrinkB=10))
plt.title(
    'Distribution of number of incidents per day', fontdict={'fontsize': 16})
plt.xlabel('Incidents')
plt.ylabel('Density')
plt.legend().remove()
st.pyplot()


st.markdown("***")
st.markdown("Crime on hourly basis plotted yearly")
data = df.groupby(['hour', 'day', 'type'],
                     as_index=False).count().iloc[:, :4]
data.rename(columns={'Dates': 'Incidents'}, inplace=True)
data = data.groupby(['hour', 'type'], as_index=False).mean()
data = data.loc[data['type'].isin(
    ['act379', 'act13', 'act279','act323','act363', 'act302'])]

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 4))
ax = sns.lineplot(x='hour', y='day', data=data, hue='type')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
plt.suptitle('Average number of incidents per hour')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot()