from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image



#obj = DeepFace.analyze(img_path = "samples/face1.jpg", actions = ['emotion'])
#print(obj)

def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv'):
	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
	img = cv2.resize(img, target_size)
	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]
	
	return img_pixels

def emotion(img):
    model = DeepFace.build_model('Emotion')

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    img = preprocess_face(img = img, target_size = (48, 48), grayscale = True)
    emotion_predictions = model.predict(img)[0,:]
    sum_of_predictions = emotion_predictions.sum()
    resp_obj = {}
    resp_obj["emotion"] = {}

    for i in range(0, len(emotion_labels)):
	    emotion_label = emotion_labels[i]
	    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
	    resp_obj["emotion"][emotion_label] = emotion_prediction
			
    resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

    return resp_obj
img = cv2.imread("samples/face1.jpg")
emotion(img)