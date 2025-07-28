# Rotates from one aspect to another; written almost exclusively by Amadea de Silva

import math as ma
import numpy as np


def to_cart(lat, lon):
    return np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])


def from_cart(point):
    lat = np.arcsin(point[2, :])
    lon = np.arctan2(point[1, :], point[0, :])
    return lat, lon


def construct_matrix(new_lat, new_lon, n_rot):
    # set up matrix -> constructed s.t. y-axis clockwise rotation -> z-axis anticlockwise rotation
    rotation = np.empty((3, 3))

    # set matrix values
    rotation[0, 0] = np.cos(new_lon) * np.cos(new_lat)
    rotation[0, 1] = -np.sin(new_lon) * np.cos(n_rot) + np.sin(n_rot) * np.sin(
        new_lat
    ) * np.cos(new_lon)
    rotation[0, 2] = -np.sin(new_lat) * np.cos(new_lon) * np.cos(n_rot) - np.sin(
        n_rot
    ) * np.sin(new_lon)

    rotation[1, 0] = np.sin(new_lon) * np.cos(new_lat)
    rotation[1, 1] = np.cos(new_lon) * np.cos(n_rot) + np.sin(n_rot) * np.sin(
        new_lat
    ) * np.sin(new_lon)
    rotation[1, 2] = -np.sin(new_lat) * np.sin(new_lon) * np.cos(n_rot) + np.sin(
        n_rot
    ) * np.cos(new_lon)

    rotation[2, 0] = np.sin(new_lat)
    rotation[2, 1] = -np.sin(n_rot) * np.cos(new_lat)
    rotation[2, 2] = np.cos(new_lat) * np.cos(n_rot)
    return rotation


def find_point(tar_lat, tar_lon, inverse):
    tar_point = to_cart(tar_lat, tar_lon)

    # apply inverse rotation matrix to target point to find in new basis
    tar_point_nb = np.matmul(inverse, tar_point)

    return from_cart(tar_point_nb)


def rev_find_point(nb_tar_lat, nb_tar_lon, rotation):
    tar_point = to_cart(nb_tar_lat, nb_tar_lon)

    # apply rotation matrix to target point to find in new basis
    tar_point_ob = np.matmul(rotation, tar_point)

    return from_cart(tar_point_ob)


def Rotate_from(lon, lat, lon_out, lat_out, rot_out):
    if lat_out != 0 or rot_out != 0:  # Use rotation matrix only where necessary
        rotation = construct_matrix(lat_out, lon_out, rot_out)
        latr, lonr = rev_find_point(lat.flatten(), lon.flatten(), rotation)
        lon = np.reshape(lonr, lon.shape)
        lat = np.reshape(latr, lat.shape)
    elif lon_out != 0:  # If only rotated by longitude, a simple frameshift can be used
        lon += lon_out
        if lon_out > 0:
            lon = np.where(lon > ma.pi, lon - 2 * ma.pi, lon)
        else:
            lon = np.where(lon < -ma.pi, lon + 2 * ma.pi, lon)

    return lon, lat


def Rotate_to(lon, lat, lon_in, lat_in, rot_in):
    if lat_in != 0 or rot_in != 0:
        inverse = np.transpose(
            construct_matrix(lat_in, lon_in, rot_in)
        )  # set inverse matrix -> rotation matrix so inverse is its transpose
        latr, lonr = find_point(lat.flatten(), lon.flatten(), inverse)
        lon = np.reshape(lonr, lon.shape)
        lat = np.reshape(latr, lat.shape)
    elif lon_in != 0:
        lon -= lon_in
        if lon_in < 0:
            lon = np.where(lon > ma.pi, lon - 2 * ma.pi, lon)
        else:
            lon = np.where(lon < -ma.pi, lon + 2 * ma.pi, lon)

    return lon, lat
