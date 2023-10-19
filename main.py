import open3d as o3d
import numpy as np


def main():

    cloud = o3d.io.read_point_cloud('PointClouds_61/0001.pcd')
    print(np.asarray(cloud.points))
    print(np.asarray(cloud.colors))
    print(np.asarray(cloud.normals))
    o3d.visualization.draw_geometries([cloud])
    

if __name__ == "__main__":
    main()