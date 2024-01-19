#proto
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

api = 'https://github.com/alexthumba/streamlitdeploymenttesting/blob/main/data.csv'
#api = 'C:/Users/alext/OneDrive/Documents/Analytics/dkcensus/PopulationStatistics/dashboard/transform/df.csv'
df = pd.read_csv(api)
df.head()

#def dataprep(df):
#Total Number of people in Denmark

#Data Prep
values_to_exclude = {'HomeCountry': ['Total'],
                    'Commune': ['Region Hovedstaden','Region Sjælland','Region Syddanmark','Region Midtjylland','Region Nordjylland'],
                    'Commune2': ['All Denmark']
                    }

# Create a condition for each column
Homecountry_condition = df['HomeCountry'].isin(values_to_exclude['HomeCountry'])
commune_condition     = df['Commune'].isin(values_to_exclude['Commune'])
commune_condition2    = df['Commune'].isin(values_to_exclude['Commune2'])

# Apply the conditions to filter the DataFrame
communedk = df[~Homecountry_condition & ~commune_condition & ~commune_condition2].reset_index(drop=True)
#5944145
#regiondk = df[~Homecountry_condition & commune_condition].reset_index(drop=True)
#regiondk = regiondk.rename(columns={'Commune': 'Region'})

#return(communedk,regiondk)

#----------- KEY METRICES
Communes_Christiansø_inc= communedk.Commune.nunique()
residents = communedk.Population.sum()
danishcitizen = communedk[communedk['HomeCountry']=='Denmark'].Population.sum()
expats =residents - danishcitizen
expats_ratio = (expats*100/residents).round(2)
men = communedk[communedk['Gender']=='Men'].Population.sum() 
women = communedk[communedk['Gender']=='Women'].Population.sum()
sex_ratio = (men*100/women).round(1)


#----------- KEY METRICES

#aggregations
#Analysis by Commune

a = communedk.groupby([(communedk['HomeCountry']. \
                        apply(lambda x: 'citizen' if x =='Denmark' else 'Indian' if x == 'India' else 'expats')),'Commune']) \
                            ['Population'].sum().reset_index(). \
                                pivot_table(index = 'Commune' ,columns='HomeCountry', values='Population', aggfunc='sum').reset_index()

a['Population'] = a['citizen']+a['expats']+a['Indian']
a['%Population_rate_per_Commune']= round(a['Population']*100/a['Population'].sum(),1)
a['%_expats_in_Commune'] = round((a['expats']+a['Indian'])*100/(a['citizen']+a['expats']+a['Indian']),1)
a.sort_values(by='Population', ascending=False,inplace=True)
a['%Population_rate_per_Commune_cumsum'] = a['%Population_rate_per_Commune'].cumsum()
a['%_expats_in_Commune_cumsum'] = a['%_expats_in_Commune'].cumsum()
a['Indians_per_expats'] = a['Indian']*100/(a['expats']+a['Indian'])
a['Indians_per_population'] = round((a['Indian'])*100/
                                     ((a['citizen']+a['expats']+a['Indian'])),1)


#-----

#Analysis by HomeCountry

b = communedk.groupby(['HomeCountry','Gender'])['Population'].sum().reset_index(). \
    pivot_table(index='HomeCountry',columns='Gender', values='Population', aggfunc='sum').reset_index()
b['Population'] = b['Men']+b['Women']
b['sex_ratio_by_originalcountry'] = round((b['Men']*100 / b['Women']),1).replace([np.inf, -np.inf,np.nan],0 )
b.sort_values(by='Population',ascending=False,inplace=True)

#-------------------------------------------------------------------------------------------------------
#Data Visualization & Dashboard

st.set_page_config(
    page_title="Danish Population Dashboard",
    page_icon="✅",
    layout="wide",
)


st.title("Danish Population Dashboard")


#Keep the space for KPI's

col1, col2, col4,col5,col6 = st.columns(5)
col1.metric(label='#Communes(Christiansø incld)', value=Communes_Christiansø_inc)
col2.metric(label='residents', value=residents)
#col3.metric(label='danishcitizen', value=danishcitizen)
col4.metric(label='No of Expats', value=expats)
col5.metric(label='Expats ratio %', value=expats_ratio)
col6.metric(label='Sex Ratio (Male/Female) %', value=sex_ratio)

# Now Trend metrices
# 1. Population  by COmmune
#2.1 Top 15 communes 
#2.2 Bottom 15 communes
#3.

#-----

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x =a['Commune'], y=a['Population']),secondary_y=False)
fig.add_trace(go.Scatter(x=a['Commune'], y=a['%Population_rate_per_Commune_cumsum']),secondary_y=True)
fig.update_layout(title_text="Communes in Denmark by population including Christiansø (2023)")
fig.update_xaxes(title_text="Commune")
fig.update_yaxes(title_text="<b></b> Population Size", secondary_y=False)
fig.update_yaxes(title_text="<b></b> %Population per Commune cumulative", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(data_frame=a[:20],x='Commune',y=['Population'],title = 'Top 20 Communes by Population')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(data_frame=a[-1:-20:-1],x='Commune',y=['Population'],title = 'Bottom 20 Communes by Population')
    st.plotly_chart(fig, use_container_width=True)


#Expats population in Denmark by Original Country (Top 20)

fig = px.bar(data_frame=b[b['HomeCountry']!='Denmark'][:40],x='HomeCountry',y=['Population'],title = 'Expats population in Denmark by Original Country (Top 40)')
st.plotly_chart(fig, use_container_width=True)


# Communes in Denmark in order of highest expats ratio
fig = px.bar(data_frame=a.sort_values(by='%_expats_in_Commune',ascending=False),x='Commune',y=['%_expats_in_Commune'],title = 'Communes in Denmark in order of highest expats ratio')
st.plotly_chart(fig, use_container_width=True)

#Communes with Indian expats population is high
fig = px.bar(data_frame=a.sort_values(by='Indians_per_expats',ascending=False),x='Commune',y=['Indians_per_expats'],title = 'Communes in Denmark in order of  Indians to expats ratio')
st.plotly_chart(fig, use_container_width=True)


#sex ratio distribution of exapats in denmark by original country

fig = px.bar(data_frame=b[b['HomeCountry']!='Denmark'].sort_values(by = 'sex_ratio_by_originalcountry',ascending=False)[:40],x='HomeCountry',y=['sex_ratio_by_originalcountry'],title = 'sex ratio (Male to Female ratio) distribution of exapats in denmark by original country (Top 40)')
st.plotly_chart(fig, use_container_width=True)


##
##
##






# -----------------------------------------------------------------------------------------------
a = communedk.groupby([(communedk['HomeCountry']. \
                        apply(lambda x: 'citizen' if x =='Denmark' else 'expats')),'Commune']) \
                            ['Population'].sum().reset_index(). \
                                pivot_table(index = 'Commune' ,columns='HomeCountry', values='Population', aggfunc='sum').reset_index() 

a['Population'] = a['citizen']+a['expats']
a['%Population_rate_per_Commune']= round(a['Population']*100/a['Population'].sum(),1)
a['%_expats_in_Commune'] = round(a['expats']*100/(a['citizen']+a['expats']),1)
a.sort_values(by='Population', ascending=False,inplace=True)  
a['%Population_rate_per_Commune_cumsum'] = a['%Population_rate_per_Commune'].cumsum()
a['%_expats_in_Commune_cumsum'] = a['%_expats_in_Commune'].cumsum()
#-----


#1 - Population by commune
# fig = px.bar(a, x="Commune", y="Population", title="Population of Denmark by Communes")
# st.plotly_chart(fig, use_container_width=True)

# fig = px.bar(a[['Commune','%_expats_in_Commune']].sort_values(by = '%_expats_in_Commune', ascending=False), \
#              x="Commune", y="%_expats_in_Commune", \
#                 title="Population across Communes")
# st.plotly_chart(fig, use_container_width=True)


#----------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x =a['Commune'], y=a['Population']),secondary_y=False)
fig.add_trace(go.Scatter(x=a['Commune'], y=a['%Population_rate_per_Commune_cumsum'],text=["Text A", "Text B"]),secondary_y=True)
fig.update_layout(title_text="Population by Commune testing")
fig.update_xaxes(title_text="Commune")
fig.update_yaxes(title_text="<b></b> Population Size", secondary_y=False)
fig.update_yaxes(title_text="<b></b> %Population per Commune cumulative", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)
#-------------------------------------------------------------------------------------------

# a = communedk.groupby([(communedk['HomeCountry']. \
#                         apply(lambda x: 'citizen' if x =='Denmark' else 'expats')),'Commune']) \
#                             ['Population'].sum().reset_index(). \
#                                 pivot_table(index = 'Commune' ,columns='HomeCountry', values='Population', aggfunc='sum').reset_index() 

# a['Population'] = a['citizen']+a['expats']
# a['%Population_rate_per_Commune']= round(a['Population']*100/a['Population'].sum(),1)
# a['%_expats_in_Commune'] = round(a['expats']*100/(a['citizen']+a['expats']),1)
# a.sort_values(by='Population', ascending=False,inplace=True)  
# a['%Population_rate_per_Commune_cumsum'] = a['%Population_rate_per_Commune'].cumsum()
# a['%_expats_in_Commune_cumsum'] = a['%_expats_in_Commune'].cumsum()