import os

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("cattap").project("_tinh_terk_kork_-y4qtj")
version = project.version(1)
dataset = version.download("yolov11")
