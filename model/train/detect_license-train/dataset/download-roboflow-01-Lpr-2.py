import os

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("cattap").project("lpr-b8uyq-er9lj")
version = project.version(2)
dataset = version.download("yolov11")
