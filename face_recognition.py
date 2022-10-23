import numpy as np
import sklearn
import pickle
import cv2
from keras.models import load_model
from PIL import Image

# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
# model_svm =  pickle.load(open('./model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
# pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary
# model_pca = pca_models['pca'] # PCA model
# mean_face_arr = pca_models['mean_face'] # Mean Face
pkl_filename = './model/faces_svm_home.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
pkl_filename = './model/output_enc_home.pkl'
with open(pkl_filename, 'rb') as file:
    output_enc = pickle.load(file)

def faceRecognitionPipeline(filename,path=True):
    dest_size = (160, 160)
    if path:
        # step-01: read image
        img = cv2.imread(filename) # BGR
    else:
        img = filename # array
    # step-02: convert into gray scale
    pixels = img.copy()
    gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # step-03: crop the face (using haar cascase classifier)

    faces = haar.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE)
    predictions = []
    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = pixels[y:y+h,x:x+w]
        image = Image.fromarray(roi)
        image = image.resize(dest_size)
         # Lây face embeding
        face_emb =  get_embedding(facenet_model, np.array(image))
        # Chuyển thành tensor
        face_emb = np.expand_dims(face_emb, axis=0)
        # Predict qua SVM
        y_hat = pickle_model.predict(face_emb)
        # # step-04: normalization (0-1)
        # roi = roi / 255.0
        # # step-05: resize images (100,100)
        # if roi.shape[1] > 100:
        #     roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        # else:
        #     roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)

        # step-06: Flattening (1x10000)
        # roi_reshape = roi_resize.reshape(1,10000)
        # # step-07: subtract with mean
        # roi_mean = roi_reshape - mean_face_arr # subtract face with mean face
        # # step-08: get eigen image (apply roi_mean to pca)
        # eigen_image = model_pca.transform(roi_mean)
        # # step-09 Eigen Image for Visualization
        # eig_img = model_pca.inverse_transform(eigen_image)
        # step-10: pass to ml model (svm) and get predictions
        # results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(face_emb)
        prob_score_max = prob_score.max()
        predict_names = output_enc.inverse_transform(y_hat)
        # step-11: generate report
        text = "%s : %d"%(predict_names[0],prob_score_max*100)
        # defining color based on results
        # if predict_names[0] == 'male':
        #     color = (255,255,0)
        # else:
        #     color = (255,0,255)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(255,255,0),-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),5)
        output = {
            'roi':roi,
            'roi':roi,
            'prediction_name':results[0],
            'score':prob_score_max
        }

        predictions.append(output)

    return img, predictions