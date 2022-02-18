from src.process_svo import SVO, Coord_System, Resolution, Unit

# Path containing your .svo files
path = "./svo_files"

# Config file and checkpoint file of your custom model
config_file = "./mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = (
    "./mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
)

custom_model = {
    "config": config_file,
    "check_pnt": checkpoint_file,
}

# The SVO class will handle everything for us
svo = SVO(
    svo_path=path + "/HD1080.svo",
    resolution=Resolution.HD1080,
    coord_system=Coord_System.ROS,
    unit_system=Unit.METER,
    custom_model=custom_model,
)

# Now save the point cloud corresponding to the .svo
svo.save_point_cloud(custom_model=True)

# We can tell which class we wanna see

# Show the result
