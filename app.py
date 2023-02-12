import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pyodbc as dbc

# Initialize Page Config
st.set_page_config(
    page_title='Reporting Dashboard', 
    page_icon=':unamused:',
    layout='wide'
)

# MSSQL DB Connection
db = {
    'servername': 'DESKTOP-M67RTHU\SQLEXPRESS',
    'database': 'GroupProjectTest'
}
conn = dbc.connect(
    'DRIVER={SQL Server};SERVER='+db['servername']+';DATABASE='+db['database']+';Trusted_Connection=yes'
)

# Read Data from CSV and append to memory
@st.cache_data
def get_data():
    df = pd.read_csv(
        filepath_or_buffer='./data/insurance.csv',
        delimiter=','
    )
    return df
df = get_data()

# Construct the Upper Main Page
title_container = st.container()
col1, mid, col2 = st.columns([2, 4, 1])
with title_container:
    with mid:
        st.title(':unamused: Insurance Company Reporting Dashboard')

# Construct the Sidebar with Filters
st.sidebar.header('Select a Filter')
gender = st.sidebar.multiselect(
    'Gender:',
    options=df['sex'].unique(),
    default=df['sex'].unique()
)
region = st.sidebar.multiselect(
    "Region:",
    options=df['region'].unique(),
    default=df['region'].unique()
)
smoker = st.sidebar.multiselect(
    "Smoker:",
    options=df['smoker'].unique(),
    default=df['smoker'].unique()
)    
age = st.sidebar.select_slider(
    "Age:",
    options=df['age'].sort_values(),
    value=30
)
bmi = st.sidebar.select_slider(
    "BMI:",
    options=df['bmi'].sort_values(),
    value=27
)

# Append Query Parameters to DataFrame
df_selection = df.query(
   "sex == @gender & region == @region & smoker == @smoker & age <= @age & bmi <= @bmi"
)

# Construct the plots
# Charges by Age Plot
charges_by_age = (
    df_selection.groupby(df['age'], group_keys=True).apply(lambda x: x)
)
fig_charges_by_age = px.bar(
    charges_by_age,
    x="age",
    y="charges",
    title="<b>Charges by Age with Query Parameters</b>",
    color_discrete_sequence=["#0083B8"] * len(charges_by_age),
    template="plotly_white"
)
fig_charges_by_age.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

# Charges by Sex Plot
charges_by_sex = (
    df_selection.groupby(df['sex'], group_keys=True).apply(lambda x: x)
)
fig_charges_by_sex = px.bar(
    charges_by_sex,
    x="sex",
    y="charges",
    title="<b>Charges by Sex with Query Parameters</b>",
    color_discrete_sequence=["#0083B8"] * len(charges_by_sex),
    template="plotly_white"
)
fig_charges_by_sex.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

#Charges by Smoke Plot
charges_by_smoke = (
    df_selection.groupby(df['smoker'], group_keys=True).apply(lambda x: x)
)
fig_charges_by_smoke = px.bar(
    charges_by_smoke,
    x="smoker",
    y="charges",
    title="<b>Charges by Smoker with Query Parameters</b>",
    color_discrete_sequence=["#0083B8"] * len(charges_by_smoke),
    template="plotly_white"
)
fig_charges_by_smoke.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

#Charges by BMI Plot
charges_by_bmi = (
    df_selection.groupby(df['bmi'], group_keys=True).apply(lambda x: x)
)
fig_charges_by_bmi = px.bar(
    charges_by_bmi,
    x="bmi",
    y="charges",
    title="<b>Charges by BMI with Query Parameters</b>",
    color_discrete_sequence=["#0083B8"] * len(charges_by_bmi),
    template="plotly_white"
)
fig_charges_by_bmi.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

# Append plots to UI
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_charges_by_age, use_container_width=True)
right_column.plotly_chart(fig_charges_by_sex, use_container_width=True)
bottom_left_column, bottom_right_column = st.columns(2)
bottom_left_column.plotly_chart(fig_charges_by_smoke, use_container_width=True)
bottom_right_column.plotly_chart(fig_charges_by_bmi, use_container_width=True)

# Input Data for Prediction
input_cointainer = st.container()
col1, mid, col2 = st.columns([1, 4, 1])

with input_cointainer:
    with mid:
        choise_smoker = st.selectbox(
            'Do you Smoke?',
            ['Yes', 'No']
        )
        choise_age = st.number_input(
            'What is your Age?',
            value=25
        )
        choise_bmi = st.number_input(
            'What is your BMI?',
            value=22
        )
if choise_smoker == 'Yes':
    choise_smoker = 1
else:
    choise_smoker = 0

# Load Data
data = pd.read_csv('./data/insurance.csv')

# Refactor Data
data['sex'].replace({'male': 0, 'female': 1}, inplace=True)
data['smoker'].replace({'yes': 1, 'no': 0}, inplace=True)
data['region'].replace({'southwest': 1, 'northwest': 2, 'northeast': 3, 'southeast': 4}, inplace=True)

# Refactor Data, keeping columns with high correlation to charges
df = pd.DataFrame(data, columns=['age', 'bmi', 'smoker'])
X = df

target = pd.DataFrame(data, columns=['charges'])
y = target['charges']

# Train and Test Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Get Input Data from UI and use them for predicting the Charges
input_data = [[choise_age, choise_bmi, choise_smoker]]

# Random Forest Regression Algorithm
tree = RandomForestRegressor(n_estimators=100, random_state=0)
tree_model = tree.fit(X_train, y_train)
prediction = tree_model.predict(X_test)

actual_prediction = tree_model.predict(input_data)

# Output Prediction and cpus to UI
output_container = st.container()
col1, mid, col2 = st.columns([1, 4, 1])

with output_container:
    with mid:
        st.write('The predicted charges are: ', str(actual_prediction).replace('[', '').replace(']', ''))
        st.write('Random Forest Regression Algorithm R2_Score is: ', str(r2_score(y_test, prediction)))
        cpu_socket = st.text_input('Please select CPU Socket:')
        df2 = pd.read_sql_query('SELECT * FROM dbo.cpus', conn)
        df2_selection = df2.query('Socket == @cpu_socket')
        if cpu_socket == '':
            st.write(df2)
        else:
            st.write(df2_selection)
        
# Hide Streamlit Bloat
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
