import time

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import cv2
import random

which_one = 0
dataset = pd.read_csv('labels.txt', header=None, sep=' ')
winlose = ""



def prediction(imgae_file):
    # Load the model
    model = load_model('keras_model.h5')



    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(imgae_file)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    print("==============嘎刀俗頭布===============")
    print(prediction[0][0])
    max_value = 0
    index = -1


    print(dataset)


    for num in prediction[0]:
        index += 1
        if (num > max_value):
            max_value = num
            which_one = index
    print("Max Probability Value", max_value, "Which_one:", which_one, "is", dataset[1][which_one])
    comp = random.randint(0, 2)
    print("電腦出：" + dataset[1][comp])

    if which_one == 0:
        if comp == 0:
            print("平手")
            winlose = "平手"
        elif comp == 1:
            print("贏惹")
            winlose = "贏惹"
        elif comp == 2:
            print("輸惹")
            winlose = "輸惹"

    elif which_one == 1:
        if comp == 0:
            print("輸惹")
            winlose = "輸惹"
        elif comp == 1:
            print("平手")
            winlose = "平手"
        elif comp == 2:
            print("贏惹")
            winlose = "贏惹"

    elif which_one == 2:
        if comp == 0:
            print("贏惹")
            inlose = "贏惹"
        elif comp == 1:
            print("輸惹")
            winlose = "輸惹"
        elif comp == 2:
            print("平手")
            winlose = "平手"


## source tutorial-env/bin/activate


cap = cv2.VideoCapture(0)

while (1):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    # 顯示圖片

    cv2.imshow('frame', frame)
    filename = 'savedImage.jpg'
    # 存圖片
    cv2.imwrite(filename, frame)
    # 若按下 q 鍵則離開迴圈

    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Filename

        time.sleep(0.03)
        prediction(filename)
        #cv2.putText(filename, dataset[1][which_one], (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    #cv2.imshow('Result', filename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
