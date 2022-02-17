from process_svo.process_svo import SVO, Coord_System, Resolution, Unit

path = "./svo_files"  # Path containing your .svo file
config_file = "./mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = (
    "./mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
)

custom_model = {
    "config": config_file,
    "check_pnt": checkpoint_file,
}

svo = SVO(
    svo_path=path + "/HD1080.svo",
    resolution=Resolution.HD1080,
    coord_system=Coord_System.ROS,
    unit_system=Unit.METER,
    custom_model=custom_model,  # pass a model here
)

svo.save_point_cloud(path, custom_model=True)
