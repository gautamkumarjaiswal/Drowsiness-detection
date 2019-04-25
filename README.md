# Drowsiness-detection
Application can detect drowsiness and raise alarm.

                                          INSTALLATION INSTRUCTION TO RUN DROWSINESS DETECTION


NOTE: WE WILL CREATE VIRTUAL ENVIRONMENT AND INSTALL ALL DEPENDENCIES IN THAT SO THAT IT WILL NOT CHANGE/AFFECT YOUR ROOT DIRECTORY, JUST BY DELETEING THIS FOLDER YOU CAN DELETE ITS EXISTANCE.

open windows 'cmd' typing cmd in search (not conda command window) as an administrator



1) Download and install python from> https://www.python.org/downloads/    (if you already have then ignore this step and check its version to be sure)

2) check python version> python -V

3) Install virtual environment> pip install virtualenv

4) test installation> virtualenv --version

5) Download folder from the drive link and extract it to 'C' drive and rename this folder to 'Drowsiness_detection', if already renamed then please ignore.
At this time if you open 'Drowsiness_detection' folder it would contains some files and folder like alarm, detect_drowsiness, shape_predictor_68_face_landmarks.dat (NOTE: You can download 'shape_predictor_68_face_landmarks.dat' file from my drive link https://drive.google.com/file/d/1PH6iDjnrC8ETzIcvySSB7Xa16ce9TOAR/view?usp=sharing)

6) Now change the directroy to 'Drowsiness_detection' folder
(at this type your command prompt window should look like C:\Drowsiness_detection>)

7) create virtual environment repositery>virtualenv env
(this will create a folder with name 'env' inside working reositery 'Drowsiness_detection')

8) Run the command to activate virtual environment>env\Scripts\activate

(after successful run, cmd window will look like '(env) C:\Drowsiness_detection>' means you have successfully created and activated virtual env and ready to work)




##### Install necessary packages ####################


9) (env) C:\Drowsiness_detection>pip install --upgrade imutils
10) (env) C:\Drowsiness_detection>pip install playsound
11) (env) C:\Drowsiness_detection>pip install pyobjc
12) (env) C:\Drowsiness_detection>pip install scipy
13) (env) C:\Drowsiness_detection>pip install dlib
14) (env) C:\Drowsiness_detection>pip install opencv-python
15) (env) C:\Drowsiness_detection>pip install dlib

##########  NOW YOU ARE READY To Test SCRIPT  ##############################




16) (env) C:\Drowsiness_detection>python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat


If you want to save alarm audio file run following command

17)(env) C:\Drowsiness_detection>python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

Reference:
[1] https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
