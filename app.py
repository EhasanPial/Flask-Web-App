from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras.preprocessing import image
import cv2

TEMPLATE_DIR = os.path.abspath('../templates')
STATIC_DIR = os.path.abspath('../static')

app = Flask(__name__)


model = load_model('my_model_96.h5')
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/read_details')
def read_details():
    return render_template('read_details.html')
@app.route('/fundus_check')
def fundus():
    return render_template('fundus_check.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
        
clahe = cv2.createCLAHE(clipLimit = 3)
def load_ben_color(image, sigmaX=10):
    
    #image = cv2.imread(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    r,image,b = cv2.split(image)
    image = clahe.apply(image)  
    #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    image = cv2.merge([r,image,b]) 
    
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    
   
    return image
def read_image(filename):
    img =  cv2.resize(filename, (128,128))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
   
    return x

@app.route('/fundus_check',methods=['GET','POST'])
def fundus_check():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
             
            
            filename = file.filename
            actual_class =  filename.split('_')[0]
            
            file_path = os.path.join('static/images', filename)
            
            file_path2 = os.path.join('static/images', "new"+filename)
            file.save(file_path2)
            img = cv2.imread(file_path2)
           
            
            img = cv2.resize(img, (512,512))
            
             
            img = load_ben_color(img,sigmaX=30) #prepressing method
           
            cv2.imwrite(file_path, img)
          
            img = read_image(img)
           
            class_prediction=model.predict(img) 
            class_prediction = np.around(class_prediction, decimals=4)
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
                retina = "Normal"
            elif classes_x == 1:
                retina = "Cataract"
            elif classes_x == 2:
                retina = "Glaucoma"
            elif classes_x == 3:
                retina = "Retina Disease"
            #'retina' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', retina = retina,prob=class_prediction, user_image = file_path
            ,actual=actual_class,user_image_actual=file_path2)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/predict',methods=['GET','POST'])
def predict_check():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
             
            
            filename = file.filename
            actual_class =  filename.split('_')[0]
             
            file_path = os.path.join('static/images', filename)
          
            file_path2 = os.path.join('static/images', "new"+filename)
            file.save(file_path2)
            img = cv2.imread(file_path2)
           
            
            img = cv2.resize(img, (512,512))
           
             
            img = load_ben_color(img,sigmaX=30) #prepressing method
            
            cv2.imwrite(file_path, img)
           
            img = read_image(img)
             
            class_prediction=model.predict(img) 
            class_prediction = np.around(class_prediction, decimals=4)
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
                retina = "Normal"
            elif classes_x == 1:
                retina = "Cataract"
            elif classes_x == 2:
                retina = "Glaucoma"
            elif classes_x == 3:
                retina = "Retina Disease"
            #'retina' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', retina = retina,prob=class_prediction, user_image = file_path
            ,actual=actual_class,user_image_actual=file_path2)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)