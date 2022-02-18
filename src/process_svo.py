import enum
import os

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

from .custom_model.model import Model
from .filtrate import filter_point_cloud
from .view.frame_viewer import Displayer


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
        self.model = Model(custom_model)

        # Configure and open camera
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = self.resolution.value
        self.init_params.camera_fps = self.fps
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = self.coord_system.value
        self.init_params.coordinate_units = self.unit_system.value
        self.init_params.set_from_svo_file(self.svo_path)
        self.zed = sl.Camera()

        self.solve_path()
        self.open()

    def solve_path(self) -> None:
        idx = self.svo_path[::-1].find("/") + 1
        path = self.svo_path[:-idx]
        if "results" not in os.listdir(path):
            os.mkdir(path + "/results")

        folder_name = self.svo_path.split("/")[-1].split(".")[0]
        if folder_name not in os.listdir(path + "/results"):
            os.mkdir(path + "/results/" + folder_name)

        self.results_path = path + "/results/" + folder_name

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

    def start_detection(self, custom: bool = False, threshold: float = 0.95) -> None:
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
        self.detection_parameters_rt.detection_confidence_threshold = threshold
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
        show_frames: bool = True,  # Whether to show frames while processing or not
        detect_objects: bool = True,  # Whether to detect objects in the frames or not
        save_pose: bool = True,  # Whether to save the camera pose in a file or not
        frame_start: int = 0,  # Set how many frames to skip
        save_measure: list = [],  # Save custom measures
        custom_model: bool = False,  # Whether to use a custom model or not
    ) -> None:
        """Saves the whole point cloud"""
        if not self.tracking_enabled:
            self.start_tracking()

        if not self.mapping_enabled:
            self.start_mapping()

        if detect_objects and not self.detect_enabled:
            self.start_detection(custom=custom_model)

        if show_frames:
            self.disp = Displayer(1280, 720)

        measures = [sl.Mat() for _ in save_measure]
        points = sl.FusedPointCloud()
        objects = sl.Objects()
        img = sl.Mat()

        step = 0

        runtime_params = sl.RuntimeParameters(
            measure3D_reference_frame=sl.REFERENCE_FRAME.WORLD,
            sensing_mode=sl.SENSING_MODE.STANDARD,  # https://github.com/stereolabs/zed-ros-wrapper/issues/497#issuecomment-558599922
            confidence_threshold=20,
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
                        key = self.disp.show(
                            img_ if "img_" in locals() else img.get_data()
                        )
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

        points.save(self.results_path + "/points", sl.MESH_FILE_FORMAT.PLY)

        if detect_objects:
            np.save(self.results_path + "/objects", np.array(self.objects))

        self.close()

    def open(self) -> None:
        """Opens the camera"""
        err = self.zed.open(self.init_params)

        if err != sl.ERROR_CODE.SUCCESS:
            print("Erro ao abrir a câmera. Verifique os parâmetros!", err)
            exit(-1)
        else:
            print("\nCâmera iniciada!")
            print("Arquivo aberto:", self.svo_path)
            print("Resolução:", self.resolution.name)
            print("FPS:", self.fps, "\n")

    def close(self) -> None:
        """Closes camera functions"""
        if self.tracking_enabled:
            self.zed.disable_positional_tracking()
            self.tracking_enabled = False

        if self.detect_enabled:
            self.zed.disable_object_detection()
            self.detect_enabled = False

        if self.mapping_enabled:
            self.zed.disable_spatial_mapping()
            self.mapping_enabled = False

        self.zed.close()

    def get_custom_objects(self, img_) -> list:
        """Predicts bounding boxes in the img and returns them as a list"""
        result = self.model.predict(img_)
        objects_in = []

        for box, conf, label in result:
            if conf < 0.99:
                continue
            tmp = sl.CustomBoxObjectData()
            tmp.unique_object_id = sl.generate_unique_id()
            tmp.probability = conf
            tmp.label = label
            tmp.bounding_box_2d = box
            tmp.is_grounded = True
            objects_in.append(tmp)

            # Draw the box in the frame
            img_ = self.disp.draw_bbox(img_, box, conf, self.model.classes[label])

        return img_, objects_in

    def plot_point_cloud(self, filter: bool = True, show_classes: list = []) -> None:
        """Plots a point cloud"""
        print("\nCarregando pontos salvos...")
        pcl = o3d.io.read_point_cloud(self.results_path + "/points.ply")
        print(pcl)

        print("\nRemovendo pontos desnecessários...")
        pcl = pcl.remove_non_finite_points()
        print(pcl)

        if filter:
            print("\nCarregando objetos...")
            objects = np.load(
                self.results_path + "/objects.npy", allow_pickle=True
            ).item()
            print("Objetos encontrados:", len(objects))

            print("\nFiltrando pontos...")
            pcl = filter_point_cloud(pcl, objects, show_classes=show_classes)

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
