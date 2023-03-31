import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations,permutations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pickle import dump
import os
import numpy as np

le = LabelEncoder()

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

def display_data(df):
    """
    Display data in dataframe format
    """
    if df is not None:
        df_style = df.style.apply(lambda x: ["background: yellow" if pd.isnull(v) else "" for v in x], axis = 1)
        st.write(df_style)


def check_duplicate_cols(df):
    """
    Check for duplicate columns based on matching column values
    """
    duplicate_cols = []
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            if (df[col1] == df[col2]).all():
                duplicate_cols.append(col2)
                df.drop(col2, axis=1, inplace=True)
    if len(duplicate_cols) > 0:
        st.warning(f"Duplicate columns found with matching values: {', '.join(duplicate_cols)}. Removed duplicate column(s).")
    else:
        st.success("No duplicate columns found with matching values.")
	

def check_single_value_cols(df):
    """
    Check for columns with a single value
    """
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
            df.drop(col, axis=1, inplace=True)
    if len(single_value_cols) > 0:
        st.warning(f"Columns with single value found: {', '.join(single_value_cols)}. Removed single value column(s).")
    else:
        st.success("No columns with single value found.")

def check_duplicate_rows(df):
    """
    Check for duplicate rows based on matching row values
    """
    duplicate_rows = df[df.duplicated()]
    if len(duplicate_rows) > 0:
        df.drop_duplicates(inplace=True)
        st.warning(f"Duplicate rows found. Removed {len(duplicate_rows)} duplicate row(s).")
    else:
        st.success("No duplicate rows found.")

        
def display_missing_values(df):
    """
    Display rows with missing values
    """
    missing_rows = df[df.isnull().any(axis=1)]
    if len(missing_rows) > 0:
        st.warning(f"Missing values found in {len(missing_rows)} row(s).")
        display_data(missing_rows)
        
    else:
        st.success("No missing values found.")
    return missing_rows
    
    
def make_bar_chart(column, title, ylabel, xlabel, y_offset=0.12, x_offset=700):
    ax = df.groupby(column).median()[['charges']].plot(
        kind='bar', figsize=(10, 6), fontsize=13, color='#4f4f4f'
    )
    ax.set_title(title, size=20, pad=30)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.get_legend().remove()
                  
    for i in ax.patches:
        ax.text(i.get_x() + x_offset, i.get_height() + y_offset, f'${str(round(i.get_height(), 2))}', fontsize=15)
    return ax
    
def show_summary_statistics(df):
    st.write("Summary Statistics:")
    st.write(df.describe())
    
def encode_binary_column(df):
    for feature in df.columns:
        if df[feature].nunique() == 2:
            unique_values = list(df[feature].unique())
            df[feature] = df[feature].apply(lambda x: 0 if x == unique_values[0] else 1)
            st.success(f"{feature} has been encoded as binary 0 and 1.")
        else:
            st.success("No Column found with2 unique.")

    
def plot_distribution(df, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=False)
    plt.xlabel(column)
    plt.ylabel("Count")
    st.pyplot()
  
def display_unique_values(data):
    st.write("Unique values:")
    unique_values = pd.DataFrame(columns=["Feature", "Unique Values"])
    for feature in data.columns:
        unique_values = unique_values.append({"Feature": feature, "Unique Values": data[feature].unique()}, ignore_index=True)
    st.table(unique_values)
      
def one_hot_encode(df, cols):
    for col in cols:
        categories = df[col].unique()
        for category in categories:
            new_col_name = col + '_' + str(category)
            df[new_col_name] = np.where(df[col] == category, 1, 0)
        df.drop(columns=[col], inplace=True)
    
def check_unique_values(df):
    unique_cols = []
    for col in df.columns:
        if df[col].nunique() == df[col].count():
            unique_cols.append(col)
    if len(unique_cols) > 0:
        st.warning("The following columns contain only unique values and will be removed:")
        st.write(unique_cols)
        df.drop(columns=unique_cols, inplace=True)
    else:
        st.success("There are no columns that contain only unique values.")
        
        
def check_val(df,col,val):
    miss = df[df[col]==val].index.to_list()
    st.write(len(miss))
    if (len(miss) <= 0):
       st.success("No value found")
    else:
       st.warning("Values found")
       st.write(df[df[col]==val])
       df[df[col]==val] = np.nan
       
  

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Machine Learning Made Easy")

    # file upload widget
    file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    X = []
    Y = []
    target=None
    # if user uploads a file
    if file is not None:
        # load data
        df = load_data(file)
        # display data
        display_data(df)

        # check for duplicate columns
        if st.button("Check for Duplicate Columns"):
            check_duplicate_cols(df)
            #display_data(df)
     

        # check for columns with a single value
        if st.button("Check for Single Value Columns"):
            check_single_value_cols(df)
            #display_data(df)
        
        if st.button("Check for Unique Columns"):
            check_unique_values(df)

        # check for duplicate rows
        if st.button("Check for Duplicate Rows"):
            check_duplicate_rows(df)
            #display_data(df)

        # check for missing values
        if st.button("Check for Missing Values"):
            missing_values=display_missing_values(df)
            if len(missing_values) > 0:
                df.dropna(inplace=True)
                st.success("Data with missing values removed")
                display_data(df)
                
        if st.button("Show Summary"):
            show_summary_statistics(df)
            
        if st.button("show all unique values"):
            display_unique_values(df)
        
        
        col = st.selectbox("Select a column:", df.columns)
        val = st.selectbox("Select a value:", df[col].unique().tolist())
        if st.button("remove perticular value"):
            check_val(df,col,val)
            
            
        if st.button("Encode binary column"):
            encode_binary_column(df)
            display_data(df)
        
        if st.button("Dtype"):
            st.write(df.dtypes)
        
        if st.button('Display'):
            display_data(df)
            
        
            
        feature_to_encode = st.selectbox("Select a categorical feature to encode:", df.select_dtypes(include="object").columns)
        if st.button("Encode categorical column"):
            #one_hot_encode(df, feature_to_encode)
            df[feature_to_encode] = le.fit_transform(df[feature_to_encode])
            st.success("encoded.")
        
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        column = st.selectbox("Select a numerical variable to plot the distribution:", numerical_columns)
        plot_distribution(df, column)
        
        
        delete_col = st.selectbox("Select a column to delete:", df.columns)
        # Add a button to delete the selected column
        if st.button("Delete Column"):
            df.drop(columns=[delete_col], inplace=True)
            st.success("Column", delete_col, "has been deleted.")
            
        problem = st.selectbox("What problem you trying to solve:", list(['regression','classfication']))
        if problem == 'classfication':
            
            # Add a button to delete the selected column
            target_col = st.selectbox("Select a target data:", df.columns)
            if st.button("label encode"):
                df[target_col] = le.fit_transform(df[target_col])
                dump(le, open('enc.pkl','wb'))
                st.success("encoded.")    
 
        title = st.text_input('ENter the file name to save')
        if st.button('save files'):
            if title:
                try:
                  
                    df.to_csv(txt, index=None)
                    st.success(f"File saved successfully as {title}")
                except FileNotFoundError:
                    st.error("Invalid file path. Please enter a valid path.")
                except Exception as e:
                    st.error("An error occurred while saving the file:", e)
            

            

if __name__ == "__main__":       
    main()

