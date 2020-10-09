from django.shortcuts import render
from django.conf import settings

# Create your views here.
from rest_framework.response import Response

from .apps import UploadimageappConfig
from .models import Cell
from .models import Result


import scipy.ndimage.filters
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.externals import joblib
import tensorflow as tf
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os, shutil

# image to features
def feature_computation(image_arr):

    image = image_arr   

    features = []
    laplacian0 = scipy.ndimage.filters.laplace(image[:,:,0])
    laplacian1 = scipy.ndimage.filters.laplace(image[:,:,1])
    laplacian2 = scipy.ndimage.filters.laplace(image[:,:,2])
  
    lap_mat0 = laplacian0
    lap_mat1 = laplacian1
    lap_mat2 = laplacian2
    
    features.append(lap_mat0.sum())
    features.append(lap_mat1.sum())
    features.append(lap_mat2.sum())
   
    image_gray = (image[:,:,0] + image[:,:,1] + image[:,:,2])/3

    gray_array = image_gray
    edges = feature.canny(gray_array,sigma = 3,low_threshold=5, high_threshold=20)
    max_val = np.amax(edges)

    
    for j in range(0,edges.shape[1]):
        for i in range(0,edges.shape[0]):
            if edges[i][j] > .1*max_val:
                edges[i][j] = 0
                if edges[i+1][j] < .1*max_val:
                    break
        for ii in range(0,edges.shape[0]):
            i = edges.shape[0] - ii -1
            if edges[i][j] > .1*max_val:
                edges[i][j] = 0
                if edges[i-1][j] < .1*max_val:
                    break
                    
    features.append(edges.sum())
    return features


def random_forest_predictor(image_arr_list):
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'uploadImageApp', 'model','radom_forest_model.pkl')
        
    ind1,ind2,ind3,ind4 = image_arr_list.shape

    predictions = []

    for i in range(0,ind1):
        image_arr = image_arr_list[i,:,:,:]

        feature_vec = feature_computation(image_arr)
        model = joblib.load(MODEL_PATH)
        probability_for_single_instance = model.predict_proba([feature_vec])
        
        predictions.append(probability_for_single_instance[0])
        
    return predictions

def runMLCode(uploaded_img_path):
    print(f"call RUN ML CODE {uploaded_img_path}")
    try:
        output_image_path = os.path.join(settings.BASE_DIR,'uploadImageApp','static','outputs')
        print(f"output_image_path {output_image_path}")
        print("call image_for_explanation ML CODE")
        image_for_explanation=[]

        im=Image.open(uploaded_img_path)
        image_for_explanation.append(im.copy())
        im.close()
        print("IMAGE closed")
        image = image_for_explanation[0]
        id1 , id2 = image.size
        print(f"image_for_explanation {len(image_for_explanation)}")

        print("image size after opened " + str(image.size))

        im_arr = np.asarray(image)
        im_arr_arr = np.asarray([im_arr])

        # Show result is the cell is Uninfected or Parasitized
        pred_str = (str(random_forest_predictor(im_arr_arr)))
        print(f"Image Prediction {pred_str}")

        sp_str = pred_str.split(', ')
        first_pred_arr = sp_str[0].split('[')
        first_pred = float(first_pred_arr[len(first_pred_arr)-1])
        print(f"first_pred {first_pred}")

        second_pred_arr = sp_str[1].split(']')
        second_pred = float(second_pred_arr[0])
        print(f"second_pred {second_pred}")

        if first_pred > second_pred:
            cell_result = "Parasitized"
        else:
            cell_result = "Uninfected"

        print(f"MALARIA_RESULT : {cell_result}")

        result = Result.objects.create(cell_condition = cell_result)
        result.cell_condition = cell_result

        print(f"cell_cond {result.cell_condition}")
        result.save();

        im_resized = np.reshape(im_arr,(id1,id2,3))
        print("print array after resized "+str(im_resized.shape))

        test_image =  im_resized
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(test_image, random_forest_predictor, batch_size = 10, num_samples=1000,top_labels=2)
        print("explanation done")

        
        # 1. output image with positive_only = True
        temp, mask = explanation.get_image_and_mask(label=0, positive_only=True, num_features=10000, hide_rest=True)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        print("1. output done")
        print(os.path.join(output_image_path , 'output1.png'))
        plt.gcf()
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.savefig(os.path.join(output_image_path , 'output1.png'))


        # 2. output image with positive_only = False
        temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10000, hide_rest=False)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.gcf()
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.savefig(os.path.join(output_image_path , 'output2.png'))
        print("2. output done")

        # 3. output image with positive_only = False
        temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10000, hide_rest=False, min_weight = 0.08)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.gcf()
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.savefig(os.path.join(output_image_path , 'output3.png'))
        print("3. output done")

        predictions = {
            'error' : '0',
            'message' : 'Successfull',
        }
        
    except Exception as e:
        print(f"EXCEPTION {e}")
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    print("OUTPUT DONe")
    
    return Response(predictions)

# Create your views here.
def index(request):

    output_image_path = os.path.join(settings.BASE_DIR,'uploadImageApp','static','outputs')
        
    for filename in os.listdir(output_image_path):
        file_path = os.path.join(output_image_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    return render(request,'index.html')


def uploadImage(request):
    print("<App>Request handling......")
    output_data = {}
    try:
        p = request.FILES['image'];
        
        print(f'request.FILES {request.FILES}')
        print(f'p {type(p)}')

        cellImage = Cell(pic = p)
        cellImage.pic.name = 'uploaded_image.png'
        print(f"cellImage.pic.name beforesave{cellImage.pic.name}")
        cellImage.save();
        print(f"cellImage.pic.name After save {cellImage.pic.name}")
        print(f"cellImage.pic path {cellImage.pic.path}")
    
        predictions = runMLCode(cellImage.pic.path)
        print(predictions)
    
        #Show uploaded image
        cellImages = Cell.objects.all();
        
        recentlyUploadedCellImage = cellImages[len(cellImages)-1].pic
        print(f" CELL IMAGE PATH : {recentlyUploadedCellImage.url}")

        #Show result
        results = Result.objects.all();

        latest_cell_result = results[len(results)-1]
        print(f"index cell res {latest_cell_result.cell_condition}")

        output_data = {
            'uploaded_pic': recentlyUploadedCellImage.url,
            'cell_condition' : latest_cell_result.cell_condition,
            'yellow_area': "Yellow area: Decision influenced pixels"
        }
    except Exception as e:
        print("EXCEPTION " + str(e))
    return render(request,'index.html', output_data)
    
