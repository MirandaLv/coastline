# coastline

This post includes code to: 
1. Download a sentinel raw imagery from copernicus (https://dataspace.copernicus.eu/).
2. Preprocessing the raw imagery:
   1. Converting the raw .jp2 imagery band to .tif
   2. Reprojecting individual sentinel band (.tif) to WGS-84.
   3. Merging the 10-m bands into a multispectral imagery.
   4. Creating imagery patches and saved into 'data/sentinel/patches' for imagery inferencing.
3. A binary water/land detection model has been trained with the Earth Surface Water Dataset ((https://zenodo.org/records/5205674)), and the weight is saved under the 'weights' folder.
4. Model prediction of the generated imagery patches, and mosaic the output imagery prediction to the raw imagery size.


# Running the code

## 1. Getting the data
- Go to geojson.io to get a study area of interest, and save the study area as a geojson file, name it boundary.geojson (a testing .geojson is in the folder for testing)
- Data retrieval/download: Run downloader.py to retrieve an imagery from sentinel that intersects with the boundary.geojson, modify the cloud cover, product type, and data collection for required data filtering.

Run this command on terminal:

`python downloader.py`

***Note: Before running this code, fill in the 'ACCESS_KEY' and 'SECRET_KEY' in the downloader.py with your own credentials.***

## 2: Extracting image patches
- Data preprocessing: Run data_preprocessing.py to Extract Sentinel band info, data reprojection, merge, clipping to AOI (the default does not clip the raw imagery tile into the AOI), and create imagery patches for model inference.

- Run this command on terminal:

`python data_preprocessing.py`

## 3. Model Training with surface water dataset
A water/land classification model is pretrained (20 epoches for testing) with a subset of data from Earth Surface Water Dataset (https://zenodo.org/records/5205674). The model weight is saved under the weights folder.

## 4. Model Inference and Mapping the result
- Model inference: Run main.py to load the model, and predict water/land with the patches created in step 2, and stitch the predicted patches together.

- Run this command on terminal within the current working directory:

`python main.py --data_path data/sentinel/patches --weights weights/trained_model.pth`


The prediction imagery is under outputs directory.