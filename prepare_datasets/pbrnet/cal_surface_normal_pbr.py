from PIL import Image
import math
import matplotlib
import numpy as np
import os
import sys
import scipy.misc

sys.path.append('../scenenet/')
import cal_surface_normal

def normalize(v):
    return v/np.linalg.norm(v)

def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel * 0.001)

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=640,pixel_height=480):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalised_pixel_to_ray_array(width=640,height=480):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

def points_in_camera_coords(depth_map,pixel_to_ray_array):
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    return camera_relative_xyz

data_root_path = '/home/chengxuz/visualmaster_relate/pbrnet/0004d52d1aeeb8ae6de39d6bd993e992'
save_path = '/home/chengxuz/visualmaster_relate/pbrnet/0004d52d1aeeb8ae6de39d6bd993e992'

if __name__ == '__main__':

    # This stores for each image pixel, the cameras 3D ray vector 
    cached_pixel_to_ray_array = normalised_pixel_to_ray_array()

    file_list = os.listdir(data_root_path)
    file_list = filter(lambda x: 'depth' in x, file_list)

    print(file_list)

    for filename in file_list:
        depth_path = os.path.join(data_root_path, filename)
        surface_normal_path = os.path.join(save_path, filename.replace('depth', 'nfromd'))
        print('Converting depth image:{0} to surface_normal image:{1}'.format(depth_path,surface_normal_path))
        depth_map = load_depth_map_in_m(str(depth_path))

         # When no depth information is available (because a ray went to
         # infinity outside of a window) depth is set to zero.  For the purpose
         # of surface normal, here we set it simply to be very far away
        depth_map[depth_map == 0.0] = 50.0

        # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
        points_in_camera = points_in_camera_coords(depth_map,cached_pixel_to_ray_array)
        surface_normals = cal_surface_normal.surface_normal(points_in_camera, 480, 640)

        # Write out surface normal image.
        img = Image.fromarray(np.uint8((surface_normals+1.0)*128.0))
        img.save(surface_normal_path)
