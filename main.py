import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utilities import read_drive_file,write_drive_file
#load data from excel file

if "data_loaded" not in st.session_state:
    df = read_drive_file("Recoveries.xlsx")
    st.session_state.data_loaded=True
    st.session_state.df=df.copy(deep=True)
else:
    df=st.session_state.df.copy(deep=True)

df.drop(['id'],axis=1,inplace=True)
df['term']=df['term'].apply(lambda x: int(x[:-7]))

#title of the app
st.title("Recoveries Data Analysis")

#create a sidebar
st.sidebar.title("Let's explore the data")
with st.sidebar:
    what_you_want_to_do={"Plot a scatter plot":1,'Find distributions':2,'Fit Machine Learning Model':3,'Using the Model':4}
    select_box=st.selectbox("What do you want to do?",what_you_want_to_do)
    select_box=what_you_want_to_do[select_box]


column_map={'Interest Rate':'int_rate','Loan Amount':'loan_amnt',"Home Ownership":'home_ownership','Annual Income':'annual_inc',"Inquiry in Last 6 months":'inq_last_6mths',"Verification Status":'verification_status',"Home Ownership":'home_ownership',
            'Verification Status':'verification_status','Revolving Balance':'revol_bal','Total Account':'total_acc','Amount Paid Off':'total_rec_prncp','Price Remaining':'princ_remaining','Recovery Amount':'recoveries','Open Accounts':'open_acc','Term':'term'}

continuous=['Loan Amount','Interest Rate','Annual Income','Open Accounts','Revolving Balance','Total Account','Amount Paid Off','Price Remaining','Recovery Amount','Inquiry in Last 6 months','term']
categorical=['Home Ownership','Verification Status']
continuous_features = [column_map[x] for x in ['Loan Amount', 'Interest Rate', 'Annual Income', 'Open Accounts',
                       'Revolving Balance', 'Total Account', 'Amount Paid Off',
                       'Price Remaining', 'Inquiry in Last 6 months', 'Term']]
categorical_features = [column_map[x] for x in ['Home Ownership', 'Verification Status']]
target_variable = column_map['Recovery Amount']

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
    x=st.selectbox("Select Column to Analyse further",column_map.keys(),index=1)
    column_x=column_map[x]
    if x in continuous:
        number_of_bins=st.slider("Select number of bins",min_value=5,max_value=50)
        fig=px.histogram(df,x=column_x,title='Distribution of {}'.format(x),nbins=number_of_bins)
        st.plotly_chart(fig,use_container_width=True)
    else:
        fig=px.pie(df,names=column_x,title='Distribution of {}'.format(x))
        st.plotly_chart(fig,use_container_width=True)
elif select_box==3:
    st.write("You want to fit machine learning model")
    st.write(df.head())
    test_size=st.slider("Select test size",min_value=0.05,max_value=0.4,step=0.01)
    df[column_map['Home Ownership']].replace({"MORTGAGE":0, 'RENT':1,'OWN':2,'OTHER':-1}, inplace=True)
    df[column_map['Verification Status']].replace({"Not Verified":0, 'Source Verified':1,'Verified':2}, inplace=True)

    X = df[continuous_features + categorical_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    st.session_state.model=model
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Use the sample method to randomly shuffle the rows
    st.write("Mean Squared Error: {}".format(mse))
    st.write("R2 Score: {}".format(r2))
    
    fig = px.histogram(y_test-y_pred, nbins=50, title='Error Distribution')
    fig.update_xaxes(title_text='Error')
    fig.update_yaxes(title_text='Frequency')
    st.plotly_chart(fig, use_container_width=True)
elif select_box==4:
    if 'model' in st.session_state:
        model=st.session_state.model
        st.write("Use the model to make predictions")
        with st.form(key='input_form'):
            loan_amnt=st.number_input("Loan Amount")
            int_rate=st.number_input("Interest Rate")
            annual_inc=st.number_input("Annual Income")
            open_acc=st.number_input("Open Accounts")
            revol_bal=st.number_input("Revolving Balance")
            total_acc=st.number_input("Total Account")
            total_rec_prncp=st.number_input("Amount Paid Off")
            princ_remaining=st.number_input("Price Remaining")
            inq_last_6mths=st.number_input("Inquiry in Last 6 months")
            term=st.number_input("Term")
            home_ownership=st.selectbox("Home Ownership",['MORTGAGE','RENT','OWN'])
            verification_status=st.selectbox("Verification Status",['Not Verified','Source Verified','Verified'])
            submitted = st.form_submit_button("Submit")
        if submitted:
            input_dict={'loan_amnt':loan_amnt,'int_rate':int_rate,'annual_inc':annual_inc,'open_acc':open_acc,'revol_bal':revol_bal,'total_acc':total_acc,'total_rec_prncp':total_rec_prncp,'princ_remaining':princ_remaining,'inq_last_6mths':inq_last_6mths,'term':term,'home_ownership':home_ownership,'verification_status':verification_status}
            input_df=pd.DataFrame([input_dict])
            st.write(input_df.head())
            input_df[column_map['Home Ownership']].replace({"MORTGAGE":0, 'RENT':1,'OWN':2}, inplace=True)
            input_df[column_map['Verification Status']].replace({"Not Verified":0, 'Source Verified':1,'Verified':2}, inplace=True)
            X = df[continuous_features + categorical_features]
            y=model.predict(input_df)
            st.write("Predicted Recovery Amount: {}".format(y))
    else:
        st.write("Please fit a model first")


    


