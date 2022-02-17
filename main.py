from process_svo.process_svo import SVO
from process_svo.process_svo import Unit
from process_svo.process_svo import Resolution
from process_svo.process_svo import Coord_System

path = "./svo_files" # Path containing your .svo file

# Pre-load a model and pass it to custom_model parameter

svo = SVO(
		svo_path=path + '/HD1080.svo',
		resolution=Resolution.HD1080,
		coord_system=Coord_System.ROS,
		unit_system=Unit.METER,
		custom_model=None, # pass a model here
	)

svo.save_point_cloud(path)