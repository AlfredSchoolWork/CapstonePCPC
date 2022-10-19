# CapstonePCPC
Dependencies: 
---
Python libraries: 
1. rawpy 
2. open-cv(cv2) 
3. numpy 
4. matplotlib
5. pytorch
6. multiprocessing
7. skikit-images

Python Version 3.9 and above. 
Lower version might work but reliability may be compromised. 
Python Version 3.5 and below will need modifications in the code due to the lack of f string functionality.

Software environment: Anaconda 

***Note that most of the dependencies must be installed before running the code to prevent any errors***
---

**Installation:**

To run the code, one can either download the entire zip file or create a fork and git clone through HTTPS or SSH.

---

***Procedures to run the program:***
To run the program, please ensure that all dependencies are installed and updated to the latest version.

Through Anaconda Prompt/Command Prompt:
***This method will only work if python is directly installed to the windows and activated by changing environment variables path or user has installed anaconda.*** 
1. After cloning or downloading the zip file, extract the file to a specific folder location.
2. Navigate to the file location using cd in the command prompt or anaconda prompt.
3. To run the demo, simply type python or python 3 user_interface.py 
4. The program should run and a picture will show up.
5. Click, drag and release to select the cropping region.
6. Press "C" to confirm the cropping region and a cropped image will appear.
7. Press any other keys to close the image and allow the program to start estimating the thickness.
8. The results will be shown in Final Decision with the estimated thickness. 

***The Decision of Pass-Fail does not solely depend on the average estimated thickness due to inconsistency of each powder coating***

***Note that each cropped region must be at least 650x650 pixels though that can be changed by modifying the code. Future revisions to the code will allow modification of the size simply by changing the optional parameter. ***

Through Spyder IDE:
1. Open the user_interface file in Spyder. 
2. To run the demo, press run.
3. The program should run and a picture will show up.
4. Click, drag and release to select the cropping region.
5. Press "C" to confirm the cropping region and a cropped image will appear.
6. Press any other keys to close the image and allow the program to start estimating the thickness.
7. The results will be shown in Final Decision with the estimated thickness. 

To apply to a new image:
Simply change the existing directory within the user_interface code to a new image directory and run it. Follow steps 1-8 or 1-7 and wait for the results.

To train new datasets:
Create new datasets using the dataPrep file. 

To train or modify existing machine learning model:
To train: Uncomment the trainNValidate command and run it.
To modify: Change the model by adding or removing layers. 

***Minimal instructions is given as it is assumed that user who want to train or modify existing machine learning model has prior knowledge on it.***

***pythonCodeForCapstone contains useful functions that can be utilised to prepare the data.***
