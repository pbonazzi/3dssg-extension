""" Script to generate the floor plans . We have defined three methods. 2 using reprojected bounding boxes ,
one point clouds """

import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import json


def save_plot(xy: np.array, filename: str):
    """ Save the plot

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    plt.figure(figsize=(5.5, 5.5), dpi=80)  # 1 in 	96 pixel (X)
    plt.plot(xy[:, 0], xy[:, 1], 's', color="black")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(filename)


def mesh_to_cp(textured_mesh: o3d.geometry.MeshBase, number_of_points: int = 50000) -> o3d.geometry.MeshBase:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return textured_mesh.sample_points_uniformly(number_of_points=number_of_points)


def cp_remove_stat_outliers(pcl: o3d.geometry.PointCloud, nb_neighbors: int = 30,
                            std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return pcl.remove_statistical_outlier(nb_neighbors, std_ratio)


def cp_points(pcl: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return np.asarray(pcl.points)


def img_points(xyz: np.array):
    """ Return array 2d with centered and scaled points

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    x, y = [i[0] for i in xyz], [i[1] for i in xyz]
    # min_x, min_y = abs(min(x)), abs(min(y))
    # max_x, max_y = max(x)+min_x, max(y)+min_y
    max_x, max_y = max(x), max(y)

    maxi = max(max_x, max_y)
    scal = int(520 / maxi)

    xy = np.zeros((len(xyz), 2), dtype=int)
    for i in range(len(xyz)):
        # xy[i][0]=int((xyz[i][0]+min_x)*scal)
        # xy[i][1]=int((xyz[i][1]+min_y)*scal)
        xy[i][0] = int(xyz[i][0] * scal)
        xy[i][1] = int(xyz[i][1] * scal)

    return xy


def create_bb_only_floor(bottom_corners_by_scan_id_dic, scan_id, floor_id_dict):
    scan = bottom_corners_by_scan_id_dic.get(scan_id)
    filename = os.path.join("bounding_boxes", "only_floor", scan_id + ".jpg")
    plt.figure(figsize=(5.5, 5.5))
    for obj_key, obj_val in scan.items():
        if obj_key in floor_id_dict.get(scan_id):
            plot_arg = 'bo-'
            plot_id = True
        else:
            continue

        x_list = [obj_val[i][0] for i in range(len(obj_val))]
        y_list = [obj_val[i][1] for i in range(len(obj_val))]
        plt.fill(x_list, y_list, "black")

    plt.axis('equal')
    plt.axis('off')
    plt.savefig(filename)


def create_bb_largest_inside(dicto, scan_id):
    scan = dicto.get(scan_id)
    filename = os.path.join("bounding_boxes", "largest_inside", scan_id + ".jpg")
    plt.figure(figsize=(5.5, 5.5))
    for obj_key, obj_val in scan.items():
        x_list = [obj_val[i][0] for i in range(len(obj_val))]
        y_list = [obj_val[i][1] for i in range(len(obj_val))]
        plt.fill(x_list, y_list, "black")

    plt.axis('equal')
    plt.axis('off')
    plt.savefig(filename)


def create_from_point_clouds(scan_id):
    path = os.path.join("render", "data", "three_scan", "scans", scan_id, "labels.instances.align.annotated.ply")
    mesh = o3d.io.read_triangle_mesh(path, True)
    mesh.compute_vertex_normals()

    pcl = mesh_to_cp(mesh)  # create a point cloud
    cl, ind = cp_remove_stat_outliers(pcl)
    xyz = cp_points(cl)
    xy = img_points(xyz)
    filename = os.path.join("input", "data", "threed_ssg", "raw", "floorplans", "results", "pointcloud2d",
                            scan_id + ".jpg")
    save_plot(xy, filename)


def main(type_of_floor_plan):
    bounding_boxes_all = json.load(open('input/data/threed_ssg/raw/jsons/boundingboxes/bounding_boxes_all.json', ))

    if type_of_floor_plan == "only_floor":
        floor_id_dict = json.load(open('input/data/threed_ssg/raw/jsons/objects/floor_id_dict.json', ))
        bottom_corners_by_scan_id_dic = json.load(
            open('input/data/threed_ssg/raw/jsons/boundingboxes/bottom_corners_by_scan_id_dic.json', ))
        for scan_id in bounding_boxes_all.keys():
            create_bb_only_floor(bottom_corners_by_scan_id_dic, scan_id, floor_id_dict)
    elif type_of_floor_plan == "point_clouds":
        for scan_id in bounding_boxes_all.keys():
            create_from_point_clouds(scan_id)
    elif type_of_floor_plan == "largest_inside":
        bbox_inside_final = json.load(open('input/data/threed_ssg/raw/jsons/boundingboxes/bbox_inside.json', ))
        for scan_id in bounding_boxes_all.keys():
            create_bb_largest_inside(bbox_inside_final, scan_id)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main("point_clouds")
