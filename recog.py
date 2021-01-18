from PIL import Image
import face_recognition
from emotion import emotion
import cv2
import numpy as np
import pyscreenshot as ImageGrab

def recog(img_file=None, image=None, debug=False):
    # Load the jpg file into a numpy array
    if img_file:
        image = face_recognition.load_image_file(img_file)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    results = { 'angry':0, 
                'disgust':0, 
                'fear':0, 
                'happy':0, 
                'sad':0, 
                'surprise':0, 
                'neutral':0
                }
    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))


        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        res = emotion(face_image)

        results[res['dominant_emotion']] += 1

        if debug:
            print(res)
            cv2.rectangle(image, (right, top), (left, bottom), (0, 255, 0), 2)
            cv2.putText(image, res['dominant_emotion'], (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if debug:
        pil_image = Image.fromarray(image)
        pil_image.show()

    return results

def screen_shot(debug=None):
    # grab fullscreen
    im = ImageGrab.grab()

    # save image file
    #im.save("fullscreen.png")
    res = recog(image=np.array(im), debug=debug)
    return res

if __name__ == '__main__':
    print(recog(img_file="samples/faces.jpg", debug=True))
    print(screen_shot(debug=True))



