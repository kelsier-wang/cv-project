import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

'''
  This is the helper function to encode the classes name to an unique id number
'''
def map_label_to_id(label): 
  return classname_id[label]

'''
  This is the helper function to decode the id number to its label in string
'''
def map_id_to_label(num):
    for label, label_id in classname_id.items():
        if label_id == num:
            return label
  
classname_id = {'xyzal_tab_5mg': 0, 
                'arthrotec_50mg_tablets': 1, 
                'sanctura_xr': 2, 
                'mycobutin_cap_150mg': 3, 
                'parnate': 4, 
                'lasix_20mg_tablets': 5, 
                'pantoprazole': 6, 
                'uroxatral_10_mg': 7, 
                'lasix_40mg_tablets': 8, 
                'soma_250_mg_tab': 9, 
                'amaryl_4mg_tablets': 10, 
                'promacta_tablets': 11, 
                'arthrotec_75mg_tablets': 12, 
                'multaq_tab_400mg': 13, 
                'cleocin_75mg_caps': 14, 
                'xanax_0.5_mg_tab': 15, 
                'requip_xl': 16, 
                'coreg': 17, 
                'avandia': 18, 
                'daypro_600mg': 19, 
                'requip': 20}
        
'''
  This is the fuction that takes testing_size (between 0 - 1), specifying the portion of all images used for testing, the rest of the images will be used for training; 
  For example, 0.5 means under each class, 50% of image will be used for testing and 50% of image will be used for traning. 
  image_width and image_height, specifying the image size. 
  The function returns a numpy array of training image (shape with default arguments: (number of training image, 512, 512, 3))
  a numpy array of testing image (shape with default arguments: (number of testing image, 512, 512, 3)), 
  a numpy array of labels corresponding to the training image (shape with default arguments: (number of training image, )), 
  and a numpy array of labels corresponding to the testing image (shape with default arguments: (number of testing image, ))
  
  To use the function in other file:
    1. import: import preprocessing
    2. call the function: preprocessing.generate_data()
'''

def generate_data(testing_size=0.5, image_width=512, image_height=512, path=None):
  if path == None:
    parent_dir = os.getcwd() 
    path = parent_dir + '/data/images'

  testing_img = []
  training_img = []
  testing_label = []
  training_label = []

  # iterate over folders in the images directory
  dirs = [x[0] for x in os.walk(path)]

  for d in dirs:
    if d[-6:] != "images":
      label = d.split('/')[-1]
      image_name = os.listdir(d)
      # spliting images into train and test randomly
      training_sample, testing_sample = train_test_split(image_name, test_size=testing_size)
      for s in training_sample:
        if s[-1] == 'g':
          image_path = d + '/' + s
          image = cv2.imread(image_path)
          image = cv2.resize(image, (image_width, image_height))
          image = np.array(image, np.float64)
          training_img.append(image)
          training_label.append(map_label_to_id(label))
        
      for s in testing_sample:
        if s[-1] == 'g':
          image_path = d + '/' + s
          image = cv2.imread(image_path)
          image = cv2.resize(image, (image_width, image_height))
          image = np.array(image, np.float64)
          testing_img.append(image)
          testing_label.append(map_label_to_id(label))
          
  training_img = np.array(training_img)
  testing_img = np.array(testing_img)
  training_label = np.array(training_label)
  testing_label = np.array(testing_label)

  print(training_img.shape, testing_img.shape, training_label.shape, testing_label.shape)
  return training_img, testing_img, training_label, testing_label

# generate_data();