import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pycaret import classification
from pycaret import regression
# Allow the user to upload a file

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, read it into a Pandas dataframe

@st.cache(allow_output_mutation=True,persist = True)
def load_data(file):
    """
    Load data from file
    """
    try:
        df = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def display_data(df,df1,y,y1):
    """
    Display data in dataframe format
    """
    if df is not None:
        st.write(df)
        st.write(df1)
        st.write(y)
        st.write(y1)
        st.write("X_train shape:", df.shape)
        st.write("X_test shape:", df1.shape)
        st.write("Y_train shape:", y.shape)
        st.write("Y_test shape:", y1.shape)


@st.cache(allow_output_mutation=True,persist = True) 
def targets(df, target):
   X = df.drop(columns=[target]).copy()
   y = df[target].copy()
   return X,y
   
@st.cache(allow_output_mutation=True,persist = True)
def train_test(X,y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test
    

def build_model(X,y,tar):
    s = classification.setup(df, target = tar, session_id = 123)
    best_model = classification.compare_models()
    metrics_df = classification.pull()
    metrics_df.to_csv('model_metrics.csv', index=False)
    classification.plot_model(best_model, plot = 'confusion_matrix', save=True)
    classification.plot_model(best_model, plot = 'auc', save=True)
    classification.save_model(best_model, 'my_first_pipeline', model_only=False)
    st.success('sucessfully created model')
    
    
def build_model_r(X,y,tar):
    s = regression.setup(df, target = tar, session_id = 123)
    best_model = regression.compare_models()
    metrics_df = regression.pull()
    metrics_df.to_csv('model_metrics.csv', index=False)
    regression.plot_model(best_model, plot = 'error', save=True)
    regression.plot_model(best_model, plot = 'residuals', save=True)
    regression.save_model(best_model, 'my_first_pipeline', model_only=False)
    st.success('sucessfully created model')
    
def scaler(cols):
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    dump(scaler, open('scalar.pkl','wb'))
    st.success("Columns have been standardized.")
 
      
if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    target = st.selectbox("Select a column to delete:", df.columns)
    X,y = targets(df,target)   
     
    
    st.write("X variables:")
    st.write(X.head())
    st.write("Y variable:")
    st.write(y.head())
    st.success("sucessfull.")
    
    X_train, X_test, Y_train, Y_test = train_test(X,y)
    
    
    
    st.success("Data has been split into training and testing sets.") 
    
    
    
    select_cols = st.multiselect("Select columns to Standerdise:", df.columns)    
    if st.button("Standerside"):
        scaler(select_cols)
            
    if st.button('diaply data'):
        display_data(X_train, X_test, Y_train, Y_test )
    
    preg = st.selectbox("select :", list(['regression','classification']))    
    if st.button('build model'):
        if preg == 'classification':
            build_model(X_train,Y_train, target)
        else:
            build_model_r(X_train,Y_train, target)
            
     
    if st.button('show metrix'):
        df = pd.read_csv('model_metrics.csv')
        st.write(df)
        
        
    if st.button('Predict'):
        aded_best_pipeline = classification.load_model('my_first_pipeline')
        predictions = classification.predict_model(aded_best_pipeline, data = X_test)
        st.write(predictions)
        st.write(Y_test)

       

