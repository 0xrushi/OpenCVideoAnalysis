import cv2
import sys
# The gender model architecture
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = '../weights/deploy_gender.prototxt'
# The gender model pre-trained weights
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = '../weights/gender_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Race Ethnicity model: https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/, https://drive.google.com/file/d/1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj/view
RACE_MODEL = '../weights/race_model_single_batch.h5'

# Represent the gender classes
GENDER_LIST = ['Male', 'Female']
# Initialize frame size
frame_width = 1280
frame_height = 720

# The model architecture
# download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_MODEL = '../weights/deploy_age.prototxt'
# The model pre-trained weights
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_PROTO = '../weights/age_net.caffemodel'
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

ROOT_FOLDER = '/mnt/hdd2/gender_detect'

IN_COLAB = 'google.colab' in sys.modules

def custom_plot(img):
    
  if IN_COLAB:
    #print('Running on CoLab')
    from google.colab.patches import cv2_imshow
    cv2_imshow(img)
  else:
    #print('Not running on CoLab')
    cv2.imshow("show", img)
    cv2.waitKey()  
    cv2.destroyAllWindows()
    