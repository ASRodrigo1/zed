import numpy as np


def is_inside_box(point, box) -> bool:
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


def filter_point_cloud(points: list, objects: dict, show_classes: list = []):
    """Filters a point cloud to show only selected objects"""
    valid = []
    points_array = np.asarray(points.points)

    for id_ in objects:

        """
        Need to learn how to extend sl.OBJECT_CLASS

        if objects[id_]["Label"] not in show_classes:
            continue
        """

        box = objects[id_]["3D Box"]

        if len(box):
            print("Filtrando o objeto:", id_, "|", objects[id_]["Label"])
            for idx in range(points_array.shape[0]):
                point = points_array[idx]
                if is_inside_box(point, box):
                    valid.append(idx)

    pcl = points.select_by_index(valid)
    return pcl
