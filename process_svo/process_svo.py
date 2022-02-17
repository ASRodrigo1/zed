import enum
import pathlib
import sys

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

from .displayer import Displayer

_parentdir = pathlib.Path("../mmdetection/").parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
# trunk-ignore(flake8/E402)
from mmdetection.mmdet.apis import inference_detector, init_detector

sys.path.remove(str(_parentdir))


class Resolution(enum.Enum):
    HD720 = sl.RESOLUTION.HD720
    HD1080 = sl.RESOLUTION.HD1080
    HD2K = sl.RESOLUTION.HD2K
    VGA = sl.RESOLUTION.VGA


class Coord_System(enum.Enum):
    STD = sl.COORDINATE_SYSTEM.IMAGE
    UNITY = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    OPENGL = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    DSMAX = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    UNREAL = sl.COORDINATE_SYSTEM.LEFT_HANDED_Z_UP
    ROS = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD


class Unit(enum.Enum):
    MILLIMETER = sl.UNIT.MILLIMETER
    CENTIMETER = sl.UNIT.CENTIMETER
    METER = sl.UNIT.METER
    INCH = sl.UNIT.INCH
    FOOT = sl.UNIT.FOOT


class Measure(enum.Enum):
    DEPTH = sl.MEASURE.DEPTH
    XYZ = sl.MEASURE.XYZ
    XYZRGBA = sl.MEASURE.XYZRGBA


class SVO(object):
    """
    Class to handle SVO filess
    """

    def __init__(
        self,
        svo_path: str,
        resolution: Resolution,
        coord_system: Coord_System,
        unit_system: Unit,
        fps: int = 30,
        custom_model=None,
    ) -> None:

        self.tracking_enabled = False
        self.mapping_enabled = False
        self.detect_enabled = False
        self.svo_path = svo_path
        self.resolution = resolution
        self.coord_system = coord_system
        self.unit_system = unit_system
        self.fps = fps

        # Start custom model
        if custom_model is not None:
            self.custom_model = init_detector(
                custom_model["config"], custom_model["check_pnt"], device="cuda:0"
            )
            self.classes = self.custom_model.CLASSES

        # Configure and open camera
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = self.resolution.value
        self.init_params.camera_fps = self.fps
        self.init_params.coordinate_system = self.coord_system.value
        self.init_params.coordinate_units = self.unit_system.value
        self.init_params.set_from_svo_file(self.svo_path)
        self.zed = sl.Camera()
        err = self.zed.open(self.init_params)

        if err != sl.ERROR_CODE.SUCCESS:
            print("Erro ao abrir a câmera. Verifique os parâmetros!", err)
            exit(-1)
        else:
            print("\nCâmera iniciada!")
            print("Arquivo aberto:", self.svo_path)
            print("Resolução:", self.resolution.name)
            print("FPS:", self.fps, "\n")

    def start_mapping(self) -> None:
        """Configure and starts spatial mapping module"""
        self.mapping_parameters = sl.SpatialMappingParameters()
        self.mapping_parameters.resolution_meter = (
            self.mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
        )
        self.mapping_parameters.range_meter = self.mapping_parameters.get_range_preset(
            sl.MAPPING_RANGE.MEDIUM
        )
        self.mapping_parameters.max_memory_usage = 6000  # 6GB
        self.mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        err = self.zed.enable_spatial_mapping(self.mapping_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Erro ao iniciar o mapeamento. Verifique os parâmetros!", err)
            exit(-1)
        else:
            print("Mapeamento iniciado!")
            self.mapping_enabled = True

    def start_tracking(self) -> None:
        """Configure and starts positional tracking module"""
        self.tracking_parameters = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(self.tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Erro ao iniciar o rastreamento. Verifique os parâmetros!", err)
            exit(-1)
        else:
            print("Rastreamento de posição iniciado!")
            self.tracking_enabled = True

    def start_detection(self, custom: bool = False) -> None:
        """Configure and starts object detection"""
        self.objects = {}
        self.detection_parameters = sl.ObjectDetectionParameters()
        self.detection_parameters.enable_tracking = True
        self.detection_parameters.enable_mask_output = True
        self.detection_parameters.detection_model = (
            sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
            if custom
            else sl.DETECTION_MODEL.MULTI_CLASS_BOX
        )
        self.detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
        self.detection_parameters_rt.detection_confidence_threshold = 80
        err = self.zed.enable_object_detection(self.detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print(
                "Erro ao iniciar a detecção de objetos. Verifique os parâmetros!", err
            )
            exit(-1)
        else:
            print("Detecção de objetos iniciada!")
            self.detect_enabled = True

    def save_point_cloud(
        self,
        path: str,
        format: str = "ply",
        show_frames: bool = True,
        detect_objects: bool = True,
        save_pose: bool = True,
        frame_start: int = 0,
        save_measure: list = [],
        custom_model: bool = False,
    ) -> None:
        """Saves the whole point cloud"""
        if not self.tracking_enabled:
            self.start_tracking()

        if not self.mapping_enabled:
            self.start_mapping()

        if detect_objects and not self.detect_enabled:
            self.start_detection()

        if show_frames:
            disp = Displayer(1280, 720)

        measures = [sl.Mat() for _ in save_measure]
        points = sl.FusedPointCloud()
        objects = sl.Objects()
        img = sl.Mat()

        step = 0

        runtime_params = sl.RuntimeParameters(
            measure3D_reference_frame=sl.REFERENCE_FRAME.WORLD,
            sensing_mode=sl.SENSING_MODE.STANDARD,  # https://github.com/stereolabs/zed-ros-wrapper/issues/497#issuecomment-558599922
            confidence_threshold=25,
        )

        print("\nIniciando processamento.")
        print("Pressione Q para interromper.\n")

        while True:

            err = self.zed.grab(runtime_params)

            if err == sl.ERROR_CODE.SUCCESS:
                if step >= frame_start:
                    self.zed.retrieve_image(img)

                    if detect_objects:
                        if custom_model:
                            img_, objects_in = self.get_custom_objects(
                                img.get_data()[:, :, :3].copy()
                            )
                            self.zed.ingest_custom_box_objects(objects_in)

                        self.zed.retrieve_objects(objects, self.detection_parameters_rt)
                        self.extract_detection(objects)

                    if show_frames:
                        key = disp.show(img_ if "img_" in locals() else img.get_data())
                        if key == ord("q"):
                            print("Processamento interrompido!")
                            cv2.destroyAllWindows()
                            break

                    for i in range(len(save_measure)):
                        self.zed.retrieve_measure(measures[i], save_measure[i])

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("\nTodos os frames foram lidos e processados.")
                cv2.destroyAllWindows()
                break
            else:
                print("\nErro durante operação! Erro:", err)
                break

            step += 1

        print("Extraindo nuvem de pontos...")
        self.zed.extract_whole_spatial_map(points)
        points.save(path + "/points", sl.MESH_FILE_FORMAT.PLY)

        if detect_objects:
            np.save(path + "/objects", np.array(self.objects))

        self.plot_point_cloud(path)

    def get_custom_objects(self, img_) -> list:
        """Predicts bounding boxes in the img and returns them as a list"""
        result = inference_detector(self.custom_model, img_)
        objects_in = []
        for indx in range(len(result)):
            for box in result[indx]:
                ax, ay, cx, cy, conf = box
                if conf < 0.9:
                    continue
                bx, by, dx, dy = cx, ay, ax, cy
                tmp = sl.CustomBoxObjectData()
                tmp.unique_object_id = sl.generate_unique_id()
                tmp.probability = conf
                tmp.label = indx
                tmp.bounding_box_2d = np.array([[ax, ay], [bx, by], [cx, cy], [dx, dy]])
                tmp.is_grounded = True
                objects_in.append(tmp)
                img_ = self.draw_bbox([ax, ay, cx, cy], img_, indx, conf)
        return img_, objects_in

    def draw_bbox(self, coords, img, indx, conf):
        """Draws a bouding box around objects"""
        ax, ay, cx, cy = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        img = cv2.rectangle(img, (ax, ay), (cx, cy), (255, 0, 0), 2)
        img = cv2.putText(
            img,
            self.classes[indx] + f": {conf:.3f}",
            (ax + 5, ay + 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        return img

    def is_inside_box(self, point, box) -> bool:
        """Checks if a point is inside a 3D box"""
        x, y, z = point

        # Check X
        if not (min([i[0] for i in box])) <= x <= max([i[0] for i in box]):
            return False

        # Check Y
        if not (min([i[1] for i in box])) <= y <= max([i[1] for i in box]):
            return False

        # Check Z
        if not (min([i[2] for i in box])) <= z <= max([i[2] for i in box]):
            return False

        return True

    def filter_point_cloud(self, points: list, objects: dict):
        """Filters a point cloud to show only selected objects"""
        valid = []
        points_array = np.asarray(points.points)

        for id_ in objects:
            box = objects[id_]["3D Box"]
            print("Filtrando o objeto:", id_, "|", objects[id_]["Label"])
            for idx in range(points_array.shape[0]):
                point = points_array[idx]
                if self.is_inside_box(point, box):
                    valid.append(idx)

        pcl = points.select_by_index(valid)
        return pcl

    def plot_point_cloud(self, path: str, filter: bool = True) -> None:
        """Plots a point cloud"""
        print("\nCarregando pontos salvos...")
        pcl = o3d.io.read_point_cloud(path + "/points.ply")
        print(pcl)

        print("\nRemovendo pontos desnecessários...")
        pcl = pcl.remove_non_finite_points()
        print(pcl)

        if filter:
            print("\nCarregando objetos...")
            objects = np.load(path + "/objects.npy", allow_pickle=True).item()

            print("\nFiltrando pontos...")
            pcl = self.filter_point_cloud(pcl, objects)

        print("\nPlotando resultado...")
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[-2, -2, -2]
        )
        o3d.visualization.draw_geometries([pcl, mesh_frame])

    def extract_detection(self, objects: list) -> None:
        """Extracts informations about the detected objects"""

        for obj in objects.object_list:
            if obj.id not in self.objects.keys():
                self.objects[obj.id] = {
                    "Position": obj.position,
                    "2D Box": obj.bounding_box_2d,
                    "3D Box": obj.bounding_box,
                    "Label": obj.label,
                    "Sub Label": obj.sublabel,
                    "Velocity": obj.velocity,
                    "Confidence": obj.confidence,
                    "Dimensions": obj.dimensions,
                }
