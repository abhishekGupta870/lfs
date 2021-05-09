import tensorflow as tf
import streamlit as st
from keras.models import load_model
import io
import time
from googlesearch import search
#from tensorflow import keras
from PIL import Image,ImageOps
import numpy as np
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
#from tensorflow import keras
#@st.cache
Xception_model=load_model('Xception.h5')
MobileNetV2_model=load_model('MobileNetV2_50_epoch.h5')
InceptionV3_model=load_model('InceptionV3.h5')
VGG16_model=load_model('VGG16.h5')
ResNet50_model= load_model("ResNet50.h5")
@st.cache
class plant_diseases_detection():
  global pred
  def page_setup():
    global pred
    st.set_page_config(page_title="Plant Disease Detection App", page_icon="icon.png", layout='centered', initial_sidebar_state='auto')
    
    st.title("Plant Diseases Detection")
    
  
    ######### -------------- Sidebarr--------------------->
    add_selectbox = st.sidebar.selectbox(
    'select the model for classification',
    ('MobileNetV2','VGG16',"ResNet50","InceptionV3",'Xception','About Data','Contact us'))
    #options=st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))

    def classes(pred):
      
      dict_classes={'Apple scab': 0, 'Apple Black rot': 1, 'Cedar apple rust': 2, 'Apple healthy': 3, 
                'Blueberry healthy': 4, 'Cherry Powdery mildew': 5, 'Cherry healthy': 6, 
                'Corn maize Cercospora leaf spot Gray leaf spot': 7, 'Corn maize Common rust': 8, 
                'Corn maize Northern Leaf Blight': 9, 'Corn maize healthy': 10, 'Grape Black_rot': 11, 
                'Grape Esca Black Measles': 12, 'Grape Leaf blight Isariopsis Leaf_Spot': 13, 'Grape healthy': 14, 
                'Orange Haunglongbing Citrus greening': 15, 'Peach Bacterial_spot': 16, 'Peach healthy': 17, 
                'Pepper bell Bacterial spot': 18, 'Pepper bellhealthy': 19, 'Potato Early blight': 20, 'Potato Late blight': 21, 
                'Potato healthy': 22, 'Raspberry healthy': 23, 'Soyabean healthy': 24, 'Squash Powdery mildew': 25, 
                'Strawberry Leaf scorch': 26, 'Strawberry healthy': 27, 'Tomato Bacterial_spot': 28, 'Tomato Early blight': 29,
                'Tomato Late blight': 30, 'Tomato Leaf Mold': 31, 'Tomato Septoria leaf spot': 32, 
                'Tomato Spider mites Two spotted spider mite': 33, 'Tomato Target Spot': 34, 
                'Tomato Yellow Leaf Curl Virus': 35, 'Tomato Tomato mosaic_virus': 36, 'Tomato healthy': 37}
      predicted_class=(list (dict_classes.keys())[list (dict_classes.values()).index(pred)])
      return predicted_class
    
    ########################################### INFORMATION ABOUT DATA #############################################################
    plant_class=['Apple','Potato','Tomato','Blueberry','Cherry','Corn Maize','Grapes','orange','Peach','Pepper','Rasberry','soyabeen','Squash','Straberry']
    disease_classes={'Apple scab': 0, 'Apple Black rot': 1, 'Cedar apple rust': 2, 'Apple healthy': 3, 
                'Blueberry healthy': 4, 'Cherry Powdery mildew': 5, 'Cherry healthy': 6, 
                'Corn maize Cercospora leaf spot Gray leaf spot': 7, 'Corn maize Common rust': 8, 
                'Corn maize Northern Leaf Blight': 9, 'Corn maize healthy': 10, 'Grape Black_rot': 11, 
                'Grape Esca Black Measles': 12, 'Grape Leaf blight Isariopsis Leaf_Spot': 13, 'Grape healthy': 14, 
                'Orange Haunglongbing Citrus greening': 15, 'Peach Bacterial_spot': 16, 'Peach healthy': 17, 
                'Pepper bell Bacterial spot': 18, 'Pepper bellhealthy': 19, 'Potato Early blight': 20, 'Potato Late blight': 21, 
                'Potato healthy': 22, 'Raspberry healthy': 23, 'Soyabean healthy': 24, 'Squash Powdery mildew': 25, 
                'Strawberry Leaf scorch': 26, 'Strawberry healthy': 27, 'Tomato Bacterial_spot': 28, 'Tomato Early blight': 29,
                'Tomato Late blight': 30, 'Tomato Leaf Mold': 31, 'Tomato Septoria leaf spot': 32, 
                'Tomato Spider mites Two spotted spider mite': 33, 'Tomato Target Spot': 34, 
                'Tomato Yellow Leaf Curl Virus': 35, 'Tomato Tomato mosaic_virus': 36, 'Tomato healthy': 37}


    if add_selectbox==('About Data'):
        
      
      model_selection=st.selectbox("Select the Model for  Details",('Data','Xception Model','ResNet50 Model','VGG16 Model','MobileNetV2 Model','InceptionV3 Model'))
      if model_selection=='Data':
        if st.button("Data Info"):
          st.info("Human society needs to increase food production by an estimated 70% by 2050 to feed an expected population size that is predicted to be over 9 billion"
                  " people. Currently, infectious diseases reduce the potential yield by an average of 40% with  many farmers in the developing world experiencing yield"
                  " losses as high as 100%. The widespread distribution of smartphones among crop growers around the world with an expected 5 billion smartphones by 2020"
                  " offers the potential of turning the smartphone into a valuable tool for diverse communities growing food. One potential application is the development"
                  " of mobile disease diagnostics through machine learning and crowdsourcing. Here we announce the release of over 50,000 expertly curated images on healthy"
                  " and infected leaves of crops plants through the existing online platform PlantVillage. We describe both the data and the platform."
                  " These data are the beginning of an on-going, crowdsourcing effort to enable computer vision approaches to help solve the problem of yield"
                  " losses in crop plants due to infectious diseases. We trained Plant village dataset with diffrent pretrained neural network to detect the"
                  " diseases in plants.")
        col4,col5,col6=st.beta_columns(3)
        with col4:
          if st.button("Dataset Info"):
            total_image=54306
            training_image=43430
            testing_image=10876
            resolution_image=224
            st.write("Total Image:", total_image)
            st.write("Training Size :",training_image)
            st.write("Testing Size :",testing_image)
            st.write("Resolution :",resolution_image,"x",resolution_image,'x',3)

        with col5:
          if st.button("Plant Class"):
            st.write("Name Of Plants used For Training")
            st.write(plant_class)
        with col6:
          if st.button("Disease Class"):
            st.write(disease_classes)

############## pending          ******************************************************************************IMP
            
      if model_selection=='ResNet50 Model':
        resnet_training_acc=99.28
        resnet_val_acc=97.45
        st.write("Training Accuracy of ResNet50 Model:",(resnet_training_acc),'%')
        st.write("Validation Accuracy of ResNet50 Model:",(resnet_val_acc),'%' )
        if st.button("Epochs Details"):
          image=Image.open("DataImages/resnet_top_10.png")
          st.image(image)
        
        col1,col2=st.beta_columns(2)
        with col1:
          if st.checkbox("Accuracy Graph"):
            image=Image.open('DataImages/resnet_acc.png')
            st.image(image)
        with col2:
          if st.checkbox("Loss Graph"):
            image=Image.open('DataImages/resnet_loss.png')
            st.image(image)
          
          

      if model_selection=='VGG16 Model':
        vgg_training_acc=95.69 
        vgg_val_acc=94.35
        st.write("Training Accuracy of VGG16 Model:",(vgg_training_acc),'%')
        st.write("Validation Accuracy of VGG16 Model:",(vgg_val_acc),'%' )
        if st.button("Epochs Details"):
          image=Image.open("DataImages/vgg16_top_10.png")
          st.image(image)
        
        col1,col2=st.beta_columns(2)
        with col1:
          if st.checkbox("Accuracy Graph"):
            image=Image.open('DataImages/vgg_acc.png')
            st.image(image)
        with col2:
          if st.checkbox("Loss Graph"):
            image=Image.open('DataImages/vgg_loss.png')
            st.image(image)
        


      if model_selection=='MobileNetV2 Model':
        mob_training_acc=97.83
        mob_val_acc=95.84
        st.write("Training Accuracy of MobileNetV2 Model:",(mob_training_acc),'%')
        st.write("Validation Accuracy of MobileNetV2 Model:",(mob_val_acc),'%' )
        if st.button("Epochs Details"):
          image=Image.open("DataImages/mobilenet_top_10.png")
          st.image(image)
          
        col1,col2=st.beta_columns(2)
        with col1:
          if st.checkbox("Accuracy Graph"):
            st.write(" MobileNetV2 Accuracy Graph:")
            image=Image.open('DataImages/mob_acc.png')
            st.image(image)
        with col2:
          if st.checkbox("Loss Graph"):
            st.write("MobileNetV2 Loss Graph:")
            image=Image.open('DataImages/mob_loss.png')
            st.image(image)
        
        

      if model_selection=='Xception Model':
        Xception_training_acc=98.23
        Xception_val_acc=97.82
        st.write("Training Accuracy of Xception Model:",(Xception_training_acc),'%')
        st.write("Validation Accuracy of Xception Model:",(Xception_val_acc),'%' )
        if st.button("Epochs Details"):
          image=Image.open("DataImages/xception_top_10.png")
          st.image(image)
        col1,col2=st.beta_columns(2)
        with col1:
          if st.checkbox("Accuracy Graph"):
            image=Image.open('DataImages/xception_acc.png')
            st.image(image)
        with col2:
          if st.checkbox("Loss Graph"):
            image=Image.open('DataImages/xception_loss.png')
            st.image(image)
        
        

      
      if model_selection=='InceptionV3 Model':
        Inception_training_acc=97.21
        Inception_val_acc=96.57
        st.write("Training Accuracy of InceptionV3 Model:",(Inception_training_acc),'%')
        st.write("Validation Accuracy of InceptionV3 Model:",(Inception_val_acc),'%')
        if st.button("Epochs Details"):
          image=Image.open("DataImages/inceptionv3_top_10.png")
          st.image(image)
        
        col1,col2=st.beta_columns(2)
        with col1:
          if st.checkbox("Accuracy Graph"):
            image=Image.open('DataImages/inception_acc.png')
            st.image(image)
        with col2:
          if st.checkbox("Loss Graph"):
            image=Image.open('DataImages/inception_loss.png')
            st.image(image)
        

    ################################################################## Contact us page        ##############################

    if add_selectbox=="Contact us":
      
      col8,col9=st.beta_columns(2)
      #with col8:
        #st.write("Sandeep Yadav")
       # image=Image.open("DataImages/Sandeep_Yadav.jpg")
        #st.image(image,caption="Sandeep Yadav")
    
        #st.markdown(""" <h1> wellcome """ "unsafe_allow_html=True")
      with col8:
        st.write("Abhishek Gupta")
        st.image(image,caption="Abhishek")


      with col19:
        st.write("Om Prakash Swami")
        st.image(image,caption="OmPrakash")
        
      
    def pesticide_c(pred):
      
      pest_classes={0:'Azoxystrobin + Difenoconazole', 1:'Copper Oxychloride+45/ kasugamy in 5/', 2:'Copper Oxychloride+45/ kasugamy in 5/', 3:'No Need of Pesticide Apple is healthy', 
                4:'No Need Pesticide Blueberry is healthy', 5:'propineb 70/wp ',6: 'No Need of Pesticide Cherry is healthy', 
                7:'Flusilazole 40/ Ec\n Dosage=300ml/acre', 8: 'Azoxystrobin 11/+ Tebuconazole 18.3/wwsc', 
                9:'Isoprothiolane 40/Ec', 10:'No Need of Pesticide Corn maize healthy', 11:'propineb 70/wp',
                12:'Azoxystrobin ', 13:'propineb 70/ wp', 14:'No Need of Pesticide Grape is healthy',15:'propineb 70/ wp',16:'copper Oxychloride + kasuamycin',
                  17: 'No Need Pesticide Peach is healthy',18: 'copper Oxychloride', 19:'No Need Pesticide Pepper is healthy', 20:'Azoxystrobin+Difenoconazole', 21:'Azoxystrobin+ Difenoconazole', 
                22:'No Need of Pesticide Potato healthy',23:'No Need Pesticide of Raspberry is healthy',24: 'No Need Pesticide Soybean is healthy',25: 'Flusilazole 40/ EC', 
                26:'propineb 70/wp', 27:'No Need of Pesticide Strawberry is healthy', 28:'Azoxystrobin + Difenoconazole', 29:'Azoxystrobin + Difenoconazole',
                30:'Azoxystrobin + Difenoconazole',31 : 'Flusilazole 40/ Es', 32:'copper Oxychloride+ kasugamycin', 
                33:'Propargite 57/ Ec',34: 'copper Oxychloride', 
                35:'propineb 70/ wp', 36:'azoxystrobin +Difenoconazole', 37:'No Need Pesticide of Tomato is healthy'}
      
      predicted_class_p=(list (pest_classes.values())[list (pest_classes.keys()).index(pred)])
      return predicted_class_p
    
  
              
  ############################################# url extraction of predicted diseases             #######################################################
    
    def info_pesticide(pesticide_name):
      for url in search(pesticide_name):
            st.write(url)
      
    
        

    

    ############# model selection ################
    if(add_selectbox=='VGG16' or add_selectbox=='MobileNetV2'or add_selectbox=='ResNet50' or add_selectbox=='InceptionV3' or add_selectbox=='Xception'):
      file_uploader=st.file_uploader('Upload cloth Image for Classification:')
      st.set_option('deprecation.showfileUploaderEncoding', False)
      if file_uploader is not None:
        image=Image.open(file_uploader)
        text_io = io.TextIOWrapper(file_uploader)
        image=image.resize((224,224))
        st.image(image,'Uploaded image:')
        
        col1,col2=st.beta_columns(2)

        def classify_image(image,model):
            #st.write("classifying......")
            #img = Image.open(image)
        
            img=image.resize((224,224))
            
   
      
            img=np.expand_dims(img,0)
            img=(img/255.0)

            preds=model.predict(img)
            pred=np.argmax(preds)
            return pred,preds
          
            
        
          
        with col1:
          st.write('Click for classify the image')
          if st.checkbox('Classify Image'):
            if(add_selectbox=='VGG16'):
              st.write("You are choosen Image classification with VGG16 model")
              pred,preds=classify_image(image,Xception_model)
              st.subheader("The Predicted Image is:")
              st.success(classes(pred))
              #st.write('Prediction probability :{:.2f}%'.format((np.max(preds)*100))
              st.subheader("Suggested Pesticide is:")
              st.info(pesticide_c(pred))
              st.balloons()
              
        
            
            if(add_selectbox=='MobileNetV2'):
                st.write("You are choosen Image classification with MobileNetV2")
                pred,preds=classify_image(image,MobileNetV2_model)
                st.subheader("The Predicted Image is:")
                st.success(classes(pred))
                #st.write('Prediction probability :{:.2f}%'.format((np.max(preds)*100))
                st.subheader("Suggested Pesticide is:")
                st.info(pesticide_c(pred))
                st.balloons()
            if(add_selectbox=='ResNet50'):
                st.write("You are choosen Image classification with ResNet50")
                pred,preds=classify_image(image,Xception_model)
                st.subheader("The Predicted Image is:")
                st.success(classes(pred))
                #st.write('Prediction probability :{:.2f}%'.format((np.max(preds)*100))
                st.subheader("Suggested Pesticide is:")
                st.info(pesticide_c(pred))
                st.balloons()
            if(add_selectbox=='Xception'):
                st.write("You are choosen Image classification with Xception")
                pred,preds=classify_image(image,Xception_model)
                st.subheader("The Predicted Image is:")
                st.success(classes(pred))
                #st.write('Prediction probability :{:.2f}%'.format((np.max(preds)*100))
                st.subheader("Suggested Pesticide is:")
                st.info(pesticide_c(pred))
                st.balloons()

            if(add_selectbox=='InceptionV3'):
                st.write("You are choosen Image classification with InceptionV3")
                pred,preds=classify_image(image,Xception_model)
                st.subheader("The Predicted Image is:")
                st.success(classes(pred))
                #st.write('Prediction probability :{:.2f}%'.format((np.max(preds)*100))
                st.subheader("Suggested Pesticide is:")
                st.info(pesticide_c(pred))
                st.balloons()
            
                
            with col2:
              healthy_list=[3,4,6,10,14,17,19,22,23,24,27,37]
              if st.checkbox('Pesticide info'):
                  
                if pred not in healthy_list:
                  pesticide_name=pesticide_c(pred)
                  info_pesticide(pesticide_name)
                else:
                  st.info("Plant is Healthy")
  
        
            
      else:
          st.write("Please select image:")
  
      
  page_setup()
plant_diseases_detection()
