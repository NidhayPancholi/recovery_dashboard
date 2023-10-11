import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utilities import read_drive_file
import statsmodels.api as sm
#load data from excel file

if "data_loaded" not in st.session_state:
    df = read_drive_file("Recoveries.xlsx")
    st.session_state.data_loaded=True
    st.session_state.df=df.copy(deep=True)
else:
    df=st.session_state.df.copy(deep=True)

#keep only recoveries less than 2k
df=df[df['recoveries']<2000]
#keep annual income less than 200k
df=df[df['annual_inc']<200000]
df['recoveries']=np.round(df['recoveries'])
df.drop(['id'],axis=1,inplace=True)
df['term']=df['term'].apply(lambda x: int(x[:-7]))

#title of the app
st.title("Recoveries Data Analysis")

#create a sidebar
st.sidebar.title("Let's explore the data")
with st.sidebar:
    what_you_want_to_do={"Plot a scatter plot":1,'Find distributions':2,'Fit Machine Learning Model':3,'Using the Model':4,'Linear Regression Analysis':5}
    select_box=st.selectbox("What do you want to do?",what_you_want_to_do)
    select_box=what_you_want_to_do[select_box]


column_map={'Interest Rate':'int_rate','Loan Amount':'loan_amnt',"Home Ownership":'home_ownership','Annual Income':'annual_inc',
            "Inquiry in Last 6 months":'inq_last_6mths',"Verification Status":'verification_status',"Home Ownership":'home_ownership',
            'Verification Status':'verification_status','Revolving Balance':'revol_bal','Total Account':'total_acc','Amount Paid Off':'total_rec_prncp',
            'Principal Remaining':'princ_remaining','Recovery Amount':'recoveries','Open Accounts':'open_acc','Term':'term'}

continuous=['Loan Amount','Interest Rate','Annual Income','Open Accounts','Revolving Balance',
'Total Account','Amount Paid Off','Principal Remaining','Inquiry in Last 6 months','Term']
categorical=['Home Ownership','Verification Status']
continuous_features = [column_map[x] for x in continuous]
categorical_features = [column_map[x] for x in categorical]
target_variable = column_map['Recovery Amount']


if select_box==1:
    x=st.selectbox("Select x-axis",continuous+['Recovery Amount'])
    y=st.selectbox("Select y-axis",continuous+['Recovery Amount'])
    color=st.selectbox("Select color",column_map.keys(),index=2)#index is used to select the default value
    column_x,column_y=column_map[x],column_map[y]
    column_color=column_map[color]
    fig=px.scatter(df,x=column_x,y=column_y,title='{} vs {}'.format(x,y),hover_data=[column_x,column_y],color=column_color)
    st.plotly_chart(fig, use_container_width=True)
elif select_box==2:
    st.write("You want to find distributions")
    x=st.selectbox("Select Column to Analyse further",column_map.keys(),index=1)
    column_x=column_map[x]
    if x in categorical:
        fig=px.pie(df,names=column_x,title='Distribution of {}'.format(x))
        st.plotly_chart(fig,use_container_width=True)
    else:
        number_of_bins=st.slider("Select number of bins",min_value=5,max_value=50)
        fig=px.histogram(df,x=column_x,title='Distribution of {}'.format(x),nbins=number_of_bins)
        st.plotly_chart(fig,use_container_width=True)
    
elif select_box==3:
    st.write("You want to fit machine learning model")
    with st.form(key='model_train_form'):
        st.write("Select the features to use for training")
        st.write("Continuous Features")
        #multiselect is used to select multiple values
        st.multiselect("Select Continuous Variable",continuous,key='selected_continuous_values')
        st.multiselect("Select Categorical Variable",categorical,key='selected_categorical_values')
        test_size=st.slider("Select test size",min_value=0.05,max_value=0.4,step=0.01)
        submitted=st.form_submit_button("Fit Model")
    if submitted:
        st.session_state.categorical_values=st.session_state.selected_categorical_values or []
        st.session_state.continuous_values=st.session_state.selected_continuous_values or []
        st.write("Model is being trained")
        df[column_map['Home Ownership']].replace({"MORTGAGE":0, 'RENT':1,'OWN':2,'OTHER':-1}, inplace=True)
        df[column_map['Verification Status']].replace({"Not Verified":0, 'Source Verified':1,'Verified':2}, inplace=True)
        cols=[column_map[x] for x in st.session_state.continuous_values+ st.session_state.categorical_values]
        X = df[cols]
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.session_state.model=model
        y_pred = model.predict(X_test)
        y_pred=np.round(y_pred)
        # Fit a Linear Regression model using statsmodels
        fig = px.histogram(abs(y_test-y_pred), nbins=50, title='Partial Residual Plot')
        fig.update_xaxes(title_text='Error')
        fig.update_yaxes(title_text='Frequency')
        st.plotly_chart(fig, use_container_width=True)

        fig=px.scatter(x=y_test,y=y_pred,title='Actual vs Predicted Recovery Amount',hover_data={"Actual Recovery Amount":y_test,"Predicted Recovery Amount":y_pred})
        #add a X=Y line
        fig.add_trace(px.line(x=[0,max(y_test)],y=[0,max(y_test)]).data[0])
        fig.update_xaxes(title_text='Actual Recovery Amount')
        fig.update_yaxes(title_text='Predicted Recovery Amount')
        st.plotly_chart(fig, use_container_width=True)

        X = sm.add_constant(X)  # Add a constant term for the intercept
        model_sm = sm.OLS(y_train, X_train).fit()

        # Get the summary of the model
        model_summary = model_sm.summary()
        st.write(model_summary)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Use the sample method to randomly shuffle the rows
        st.write("Mean Squared Error: {}".format(mse))
        st.write("R2 Score: {}".format(r2))
        
elif select_box==4:
    if 'model' in st.session_state:
        model=st.session_state.model
        st.write("Use the model to make predictions")
        input_dict={}
        with st.form(key='input_form'):
            if 'Loan Amount' in st.session_state.continuous_values:
                input_dict[column_map['Loan Amount']]=st.number_input("Loan Amount",min_value=min(df['loan_amnt']),max_value=max(df['loan_amnt']))
            if 'Interest Rate' in st.session_state.continuous_values:
                input_dict[column_map['Interest Rate']]=st.number_input("Interest Rate",min_value=min(df['int_rate']),max_value=max(df['int_rate']))
            if 'Annual Income' in st.session_state.continuous_values:
                input_dict[column_map['Annual Income']]=st.number_input("Annual Income",min_value=min(df['annual_inc']),max_value=max(df['annual_inc']))
            if 'Open Accounts' in st.session_state.continuous_values:
                input_dict[column_map['Open Accounts']]=st.number_input("Open Accounts",min_value=min(df['open_acc']),max_value=max(df['open_acc']))
            if 'Revolving Balance' in st.session_state.continuous_values:
                input_dict[column_map['Revolving Balance']]=st.number_input("Revolving Balance",min_value=min(df['revol_bal']),max_value=max(df['revol_bal']))
            if 'Total Account' in st.session_state.continuous_values:
                input_dict[column_map['Total Account']]=st.number_input("Total Account",min_value=min(df['total_acc']),max_value=max(df['total_acc']))
            if 'Amount Paid Off' in st.session_state.continuous_values:
                input_dict[column_map['Amount Paid Off']]=st.number_input("Amount Paid Off",min_value=min(df['total_rec_prncp']),max_value=max(df['total_rec_prncp']))
            if 'Principal Remaining' in st.session_state.continuous_values:
                input_dict[column_map['Principal Remaining']]=st.number_input("Principal Remaining",min_value=min(df['princ_remaining']),max_value=max(df['princ_remaining']))
            if 'Inquiry in Last 6 months' in st.session_state.continuous_values:
                input_dict[column_map['Inquiry in Last 6 months']]=st.number_input("Inquiry in Last 6 months",min_value=min(df['inq_last_6mths']),max_value=max(df['inq_last_6mths']))
            if 'Term' in st.session_state.continuous_values:
                input_dict[column_map['Term']]=st.number_input("Term",min_value=min(df['term']),max_value=max(df['term']))
            if 'Home Ownership' in st.session_state.categorical_values:
                input_dict[column_map['Home Ownership']]=st.selectbox("Home Ownership",['MORTGAGE','RENT','OWN'])
            if 'Verification Status' in st.session_state.categorical_values:
                input_dict[column_map['Verification Status']]=st.selectbox("Verification Status",['Not Verified','Source Verified','Verified'])
            submitted = st.form_submit_button("Submit")
        if submitted:
            input_df=pd.DataFrame([input_dict])
            st.write(input_df.head())
            if column_map["Home Ownership"] in input_dict:
                input_df[column_map['Home Ownership']].replace({"MORTGAGE":0, 'RENT':1,'OWN':2}, inplace=True)
            if column_map['Verification Status'] in input_dict:
                input_df[column_map['Verification Status']].replace({"Not Verified":0, 'Source Verified':1,'Verified':2}, inplace=True)
            cols=[column_map[x] for x in st.session_state.continuous_values+ st.session_state.categorical_values]
            
            X = input_df[cols]
            y=model.predict(X)
            st.write("Predicted Recovery Amount: {}".format(y))
    else:
        st.write("Please fit a model first")


elif select_box==5:
    x=st.selectbox("Select Predictor",continuous)
    facet_col=st.selectbox("Select Column to split on",categorical+['None'],index=1)#index is used to select the default value
    facet_col=column_map[facet_col] if facet_col!='None' else None
    y='Recovery Amount'
    column_x,column_y=column_map[x],column_map[y]
    fig=px.scatter(df,x=column_x,y=column_y,title='{} vs {}'.format(x,y),hover_data=[column_x,column_y],trendline='ols',facet_col=facet_col)
    results=px.get_trendline_results(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write(results.px_fit_results.iloc[0].summary())

    


