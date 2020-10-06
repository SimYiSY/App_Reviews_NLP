import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the models to predict on the data 
pickle_in_1 = open('./model_pickles/Best_Model_P1.sav', 'rb') 
classifier_1 = pickle.load(pickle_in_1)

pickle_in_2 = open('./model_pickles/Best_Model_BadReview_P2.sav', 'rb') 
classifier_2 = pickle.load(pickle_in_2) 

pickle_in_3 = open('./model_pickles/Best_Model_GoodReview_P3.sav', 'rb') 
classifier_3 = pickle.load(pickle_in_3) 
  
def welcome(): 
    return 'welcome all'
  
# defining the function which will make the prediction using  
# the data which the user inputs 
def prediction(reviews):   
   
    prediction = classifier_1.predict( 
        [reviews]) 
    print(prediction) 
    return prediction

def prediction_bad(reviews):   
   
    prediction = classifier_2.predict( 
        [reviews]) 
    print(prediction) 
    return prediction

def prediction_good(reviews):   
   
    prediction = classifier_3.predict( 
        [reviews]) 
    print(prediction) 
    return prediction
      
  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("Shopping Apps Reviews Analysis") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">App Store Reviews Classifier ML App </h1> 
    </div> 
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    reviews = st.text_input("Review", "") 
    result = ""
    final ="" 
      
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
    if st.button("Predict"): 
        result = prediction(reviews)
        if result == 1:
            result = 'Good'
            final = prediction_good(reviews)
        else:
            result = 'Bad'
            final = prediction_bad(reviews)
            
    st.success('The review rating is {}'.format(result))     
    st.success('The category of the review is {}'.format(final)) 
     
if __name__=='__main__': 
    main() 



