import streamlit as st
import pandas as pd
import plotly.express as px

#load data from excel file
df = pd.read_excel("Recoveries.xlsx")

#title of the app
st.title("Recoveries Data Analysis")
st.write(df.head())
#create a sidebar
st.sidebar.title("Let's explore the data")
with st.sidebar:
    what_you_want_to_do={"Plot a scatter plot":1,'Find distributions':2,'Find Statistics':3,'Fit Machine Learning Model':4}
    select_box=st.selectbox("What do you want to do?",what_you_want_to_do)
    select_box=what_you_want_to_do[select_box]


column_map={'Interest Rate':'int_rate','Loan Amount':'loan_amnt',"Home Ownership":'home_ownership','Annual Income':'annual_inc',"Inquiry in Last 6 months":'inq_last_6mths',"Verification Status":'verification_status',"Home Ownership":'home_ownership',
            'Verification Status':'verification_status','Revolving Balance':'revol_bal','Total Account':'total_acc','Amount Paid Off':'total_rec_prncp','Price Remaining':'princ_remining','Recovery Amount':'recoveries','Open Accounts':'open_acc'}
st.write(column_map)
continuous=['Loan Amount','Interest Rate','Annual Income','Open Accounts','Revolving Balance','Total Account','Amount Paid Off','Price Remaining','Recovery Amount']


if select_box==1:
    x=st.selectbox("Select x-axis",continuous)
    y=st.selectbox("Select y-axis",continuous)
    color=st.selectbox("Select color",column_map.keys(),index=2)#index is used to select the default value
    column_x,column_y=column_map[x],column_map[y]
    column_color=column_map[color]
    fig=px.scatter(df,x=column_x,y=column_y,title='{} vs {}'.format(x,y),hover_data=[column_x,column_y],color=column_color)
    st.plotly_chart(fig, use_container_width=True)
elif select_box==2:
    st.write("You want to find distributions")
elif select_box==3:
    st.write("You want to find statistics")
elif select_box==4:
    st.write("You want to fit machine learning model")
else:
    st.write("You want to do nothing")



