# heartrate-script
I was contacted by an investor who has been financing a project involved in the health industry, whose objective is to offer a remote medical service.
At the beginning, this entrepreneur needed a prototype that could work as a viable product and thus present it to other investors with more resources to continue the development in depth.
I was responsible for developing a module consisting of:
A software that receives facial video as input to process the frames and predict the patient's heart rate signal.

Tech stack: Python, tensorflow, pandas, jeancv, opencv, scikit-learn

How to use it:

* Go to the root folder
* Run the python command main.py --video "[video local path]"

In case of detecting a face in the video, the script will return a message like this

![alt text](https://github.com/MaximilianoAlarcon/heartrate-script/blob/master/success.png?raw=true)
