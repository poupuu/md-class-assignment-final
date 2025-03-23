#fix encode function
#manual mapping buat target variable

import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler 

#load .pkl
label_encoder = joblib.load("label_encoder.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
robust_scaler = joblib.load("robust_scaler-2.pkl")
model = joblib.load("fine_tuned_model.pkl")

def input_user_to_df(input):
    data = [input]
    df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
    return df

def encode(input_df, label_encoders):
    for column in input_df.columns:
        if input_df[column].dtype == 'object' or input_df[column].dtype.name == 'category':
            input_df[column] = label_encoders[column].transform(input_df[column])
    return input_df

def scaling(input_df, robust_scaler):
    for column in input_df.columns:
        if input_df[column].dtype == 'int64' or input_df[column].dtype.name == 'float64':
            input_df[column] = robust_scaler[column].transform(input_df[column])
    return input_df

def predict_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0] if len(prediction) > 0 else prediction

#main
def main():
    st.title("Machine Learning App")
    st.info("This app will predict your obesity level!")
    
    #raw data
    with st.expander("**Data**"):
        st.write("This is raw data")
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
        df

        st.write("**X**")
        input_df = df.drop("NObeyesdad", axis=1)
        input_df
        
        st.write("y")
        output_df = df["NObeyesdad"]
        output_df

    #data visualization
    with st.expander('**Data Visualization**'):
        st.scatter_chart(data=df, x = 'Height', y = 'Weight', color='NObeyesdad')

    #input data
    #numerical
    Age = st.slider('Age', min_value = 14, max_value = 61, value = 24)
    Height = st.slider('Height', min_value = 1.45, max_value = 1.98, value = 1.7)
    Weight = st.slider('Weight', min_value = 39, max_value = 173, value = 86)
    FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
    NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
    CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
    FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
    TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)
    
    #categorical
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    family_history_with_overweight = st.selectbox('Family history with overweight', ('yes', 'no'))
    FAVC = st.selectbox('FAVC', ('yes', 'no'))
    CAEC = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'no'))
    SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
    SCC = st.selectbox('SCC', ('yes', 'no'))
    CALC = st.selectbox('CALC', ('Sometimes', 'no', 'Frequently', 'Always'))
    MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))
    
    #input data to df
    #numerical
    
    
    #categorical
    # numerical_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"] #configurable
    # categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"] #configurable
    
    user_input = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"]
    df = input_user_to_df(user_input)
    st.write('Data input by user')
    df
    user_input = encode(user_input, label_encoder)
    user_input



    #preprocess input data
    # df = target_encode(output_df, target_encoder)
    # df = feature_encode(df)
    
    # input_df = input_user_to_df(user_input)
    
    
#     df = standard_scaling(df, standard_scaler, "Height")
#     df = robust_scaling(df, robust_scaler)
#     prediction = predict_model(model, df)
    
#     #preprocess input data
#     # df = encode(df, target_encoder, label_encoder)
#     # df = scaling(df, standard_scaler, robust_scaler)
#     # prediction = predict_model(model, df)
    
#     # #preprocess user input
#     # user_input_scaled = preprocess_input(user_input, label_encoders, standard_scaler, robust_scaler)
    
#     #predict probabilities
#     pred = model.predict_proba(df)
#     df_pred = pd.DataFrame(prediction_proba)
#     df_pred.columns = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
#     df_pred.rename(columns={0: 'Insufficient Weight', 
#                             1:'Normal Weight', 
#                             2: 'Overweight Level I', 
#                             3: 'Overweight Level II', 
#                             4:'Obesity Type I', 
#                             5:'Obesity Type II', 
#                             6: 'Obesity Type III'})


#     st.write("Obesity Prediction")
#     df_pred
#     st.write("The predicted output is: ", prediction)

if __name__ == "__main__":
    main()
