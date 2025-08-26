# Monitoring on-street parking spots (parkometer) using a static mounted street camera
This project aims to utilize yolo model to detect cars parked on the street spaces using a static camera.
Firstly, the parking spots should be specified by the user as polygons. Secondly, the yolov11 model from ultralytics is soley
used to detect cars (the model is run on a small custom dataset). Finally, using the polygon determined by the user during the
first phase, the status of parking spots on the street will be updated to either occupied or unoccupied in real time.
Note: The state of parking spots will be renewed every two seconds.
## Video Demo

https://github.com/user-attachments/assets/acdb9810-3833-48ff-ad4c-e07050fc64ee

## modules
1. The `data_split.py` file is used to split data into train and validation folders based on how the yolo models should be formatted.
(user arguments should be passed, such as train_ratio: 0.9 for instance)
2. The `data_prep.py` file is used to take snapshots of video files (multiple video fiels can be parsed in a single run)
3. The `preprocess.py` file is used to specify parking spots(polygons). In the end, spots are shown to the user, and they will be
saved within a json file.
4. The `detection.ipynb` notebook is used to train the newest yolo version on a small custom dataset that was prepared.
5. In `main.py` the final stage of this project, the status of predetermined parking spots is updated every two seconds.
