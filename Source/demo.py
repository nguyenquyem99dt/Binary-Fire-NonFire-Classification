import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options
    parser.add_argument('--model', type=str, 
    help = 'Path to model')

    parser.add_argument('--input', type=str,
    help = 'Path to input')

    parser.add_argument('--image', default=False,
    help='Image detection mode')

    FLAGS = parser.parse_args()

    # loading the stored model from file
    model=load_model(FLAGS.model)

    IMG_SIZE = 64

    # image mode
    if FLAGS.image:
        try:
            image = cv2.imread(FLAGS.input) # read image
            orig = image.copy() # copy image
            
        except:
            print('Can not read input image. Please check path again!')
        
        else:            
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # resize image to fit with model 
            image = image.astype("float") / 255.0 # convert to float
            image = img_to_array(image) # convert image to array
            image = np.expand_dims(image, axis=0) # expend dims
                
            fire_prob = model.predict(image)[0][0] * 100 # Get predict probability of fire

            label = "Fire/Smoke Probability: " + str(fire_prob)
            cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 6) # put probability of fire
            cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2) # put probability of fire
            
            cv2.putText(orig, 'Press any key to exit!', (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 240), 2)
            cv2.imwrite('prediction.jpg', orig) # save output image

            cv2.imshow("Output", orig) # show output image
            cv2.waitKey(0) # wait key
            cv2.destroyAllWindows()

    # video mode
    else: 
        cap = cv2.VideoCapture(FLAGS.input)
        time.sleep(2)

        if cap.isOpened(): # try to get the first frame
            rval, frame = cap.read()
        else:
            rval = False

        while(1):
            rval, image = cap.read()

            if rval==True:
                orig = image.copy() # copy image
            
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # resize image to fit with model
                image = image.astype("float") / 255.0 # convert to float type
                image = img_to_array(image) # convert image to array
                image = np.expand_dims(image, axis=0) # expand dims
                
                tic = time.time() # Start time
                fire_prob = model.predict(image)[0][0] * 100 # Get probability of fire
                toc = time.time() # End time

                fps = 1 / np.float64(toc - tic) # Calculate fps

                # print some information on console
                print("Time taken = ", toc - tic)
                print("FPS: ", fps)
                print("Fire/Smoke Probability: ", fire_prob)
                print("Predictions: ", model.predict(image))
                print('Image shape:', image.shape)
                print('--------------------------------')

                label = "Fire/Smoke Probability: " + str(fire_prob)

                cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 6)
                cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2) # put fire probability to output image
                cv2.putText(orig, 'FPS: '+ str(fps), (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 240, 0), 2) # put fps
                cv2.putText(orig, 'Press ESC to exit!', (10, 60),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 240), 2)

                cv2.imshow("Output", orig) # show output image
                key = cv2.waitKey(1) # wait least 1ms

                if key == 27: # exit on ESC
                    break
            elif rval==False: # If no frame to read -> break
                    print('No frame to read!')
                    break
    
        cap.release()
        cv2.destroyAllWindows()

    
