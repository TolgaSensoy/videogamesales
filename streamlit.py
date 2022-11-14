import streamlit as st
import pandas as pd 
import shap
import matplotlib.pyplot as plt
from joblib import load

st.write("""# Video game Global Sales Prediction \n""")

df = pd.read_csv('csvforst.csv', index_col = 'Rank')
df.head()

X = df.drop(['Platform', 'Genre', 'Publisher', 'Global_Sales'], axis = 1)
Y = df['Global_Sales']
df.Year = df.Year.astype('int')


def user_input_features(): 
    st.sidebar.header('Specify Input Parameters')
    Platform = st.sidebar.selectbox("Platform", options = list(X['Platform transformed'].unique()), format_func = lambda x : df['Platform'].unique()[x])
    Publisher = st.sidebar.selectbox("Publisher", options = list(X['Publisher transformed'].unique()), format_func = lambda x : df['Publisher'].unique()[x])
    Genre = st.sidebar.selectbox("Genre", options = list(X['Genre transformed'].unique()), format_func = lambda x : df['Genre'].unique()[x])
    Year = st.sidebar.selectbox("Year", options = list(X['Year'].unique()))
    data = {"Platform" : Platform,
            "Publisher" : Publisher,
            "Genre" : Genre,
            "Year" : Year}
    features = pd.DataFrame(data, index = [0])
    return features

df2 = user_input_features()



st.header('Specified Input parameters')
st.write(df2)
st.write('---')

model = load('BestRF.joblib')
model.fit(X, Y)
prediction = model.predict(df2)

st.header('Prediction of Gaming Sales')
st.markdown("(in millions)")
st.write(prediction)
st.write('---')



shap_values = shap.TreeExplainer(model).shap_values(X)
fig1 = plt.figure() 
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X.columns)
st.write('---')

st.pyplot(fig1)
    

fig2 = plt.figure()
    
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X.columns, plot_type="bar",)
plt.xlabel('average impact on model output magnitude')
st.pyplot(fig2)