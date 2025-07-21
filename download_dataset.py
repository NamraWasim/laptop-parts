from roboflow import Roboflow

rf = Roboflow(api_key="aHWujFAEiDCuQkPYajSL")
project = rf.workspace("levon-vanyan-tev1k").project("laptop-parts")
version = project.version(1)
dataset = version.download("yolov8")
