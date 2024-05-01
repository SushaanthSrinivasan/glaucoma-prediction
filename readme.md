# Glaucoma Prediction

## URL/Source for Dataset

DRISHTI-GS: https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation

ACRIMA: https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135

Zenodo OCT Images: https://zenodo.org/record/1481223#.Y20g3XbMIuV

## Software Requirements:

- Tensorflow
- Matplotlib
- Keras
- Flask
- Node JS
- Next JS

Hardware Requirements:

- Systems with good GPUs (Nvidia RTX cards) are recommended for good performance
- Systems with CPU (average performance)
- Google Colaboratory (good performance)

Source Code Execution Instructions:

## Running the Mass Screening tool

1. Change into the web-app directory.
2. Open 1 terminal inside the frontend folder.
3. Type "npm install" and execute.
4. The type "npm run dev". This will start the frontend server on localhost.
5. Open another terminal inside the backend folder.
6. Type "flask run" and execute. The backend server will start on localhost.

## Training of Few Shot Learning:

1. Install ananconda and create a new conda environment using "conda create -n <env-name> python=3.9"
2. Install all the software requiremnts mentioned in this file using "pip install <module-name>"
3. Open the fsl-vgg Ipython notebook.
4. Select the appropriate conda environment.
5. Open the notebook as a Jupyter Notebook in localhost and execute each cell consecutively.
6. This will train the model.
7. After saving the model, open the testing Ipython notebook.
8. Use the name for the saved model and load it here.
9. Execute the cells and save the output metrics.

## Training of U-Net:

1. Install ananconda and create a new conda environment using "conda create -n <env-name> python=3.9"
2. Install all the software requiremnts mentioned in this file using "pip install <module-name>"
3. Open the Final_U-net Ipython notebook.
4. Select the appropriate conda environment.
5. Open the notebook as a Jupyter Notebook in localhost and execute each cell consecutively.
6. This will train the model.
7. After saving the model, load it into the same Ipython notebook.
8. Execute the cells and save the output metrics.
9. The test.py file contains the code for calcualting the adaptive binarizer.
10. That code can be used to calculate the optimal binarizer for each image.

## Pre-trained model:

1. Install ananconda and create a new conda environment using "conda create -n <env-name> python=3.9"
2. Install all the software requiremnts mentioned in this file using "pip install <module-name>"
3. Open model.py file.
4. Select the appropriate conda environment.
5. Run the python script to save the probabilities in a CSV file.
6. This will train the model.
7. After saving the model, run analyze_model.py.
8. Execute the file and save the output metrics.
