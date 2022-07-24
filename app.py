import streamlit as st
from PIL import Image
import cv2
import torch
from PIL import Image
import pandas as pd
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import cv2
#from utils.plots import plot_one_box

@st.cache(suppress_st_warning=True)
def model():
  model = torch.hub.load('/content/drive/MyDrive/projects/Artenal/yolov5', 'custom', source='local', path = '/content/drive/MyDrive/projects/Artenal/yolov5/yolov5s.pt', force_reload = True)
  return model

st.write("### Apple and Orange Detector")
st.caption('Created by, *Lomesh Soni* :sunglasses:')
st.write("Github link [link](https://github.com/Lomesh2000?tab=repositories)")


uploaded_file = st.file_uploader("Choose an image of fruits....", type="jpg")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    #image=image.resize((200,150))

    img=image.save('img.jpg')
    img=cv2.imread('img.jpg')

    st.image(image, caption='Uploaded Fruits Image.', use_column_width=True,width=50)

    st.write("Detecting Number of Oranges and Apples........")


    #model = torch.hub.load('/content/drive/MyDrive/projects/Artenal/yolov5', 'custom', source='local', path = '/content/drive/MyDrive/projects/Artenal/yolov5/yolov5s.pt', force_reload = True)

    yolo_model=model()
    results = yolo_model(image, size=640)

    
  
    df=results.pandas().xyxy[0].name.value_counts().reset_index()
    
    cordinates=results.xyxy[0].numpy()

    cord , label = [cor[0:4] for cor in cordinates] ,[cor[4:6] for cor in cordinates]

    #image=cv2.imread(uploaded_file)
    for i,cor in enumerate(cord):
      cv2.rectangle(img,(int(cor[0]),int(cor[1])),(int(cor[2]),int(cor[3])),(0,255,0),1)
      
      if int(label[i][1])==47:
        string='Apple'
      else:
        string="orange"

      cv2.putText(img ,string  , (int(cor[0]),int(cor[1])) ,  cv2.FONT_HERSHEY_COMPLEX , 0.5 ,(255, 0, 0) ,1)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st.image(img, caption='Apples and Oranges detected.', use_column_width=True,width=50)
    #print(cordinates)

    df.columns=['Fruit','Quantity']

    

    #st.dataframe(df,200,400)
    st.table(df)
    st.write("Done")
