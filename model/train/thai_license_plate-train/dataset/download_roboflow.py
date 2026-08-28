import os

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("cattap").project("thai-license-plate-character-recognition-orhmi")
version = project.version(1)
dataset = version.download("yolov11")
