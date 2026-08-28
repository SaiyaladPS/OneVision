import os

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("mona-f3tol").project("plate-detect-gmqpu")
version = project.version(1)
dataset = version.download("yolov11")
