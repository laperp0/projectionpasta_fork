import math as ma
import os

import numpy as np
from PIL import Image

import projections_but_new.collignon_quincuncial as cq
import projections_but_new.petroff_quincuncial as pq
from lib.conversion import Main
from lib.projection_utilities import (
    Circ_vis,
    Def_int,
    Def_pres,
    Def_typ,
    Def_vis,
    From_pol,
    Hem_coords,
    Hem_pos,
    Hem_pres,
    Hem_vis,
    Hemfull_typ,
    Hemonly_typ,
    HemonlyIter_typ,
    Inv_coords,
    Iter_typ,
    No_coords_int,
    No_pos_int,
    Onehem_coords,
    Rect_int,
    To_pol,
)

# Increases potential map size to 1 million x 1 million pixels, which is probably more than anyone's RAM could actually handle
Image.MAX_IMAGE_PIXELS = 1e12

# Remapping functions for each projection:
# coords determines (lon,lat) coordinates for a given (x,y) position on the map
# pos determines (x,y) position for given (lon,lat) coordinates
# vis determines if an (x,y) position on the output should be shown on non-rectangular maps
# pres determines if an (x,y) position is present on the input
# typ incorporates extra information required for specific projection types
# rat is the map width/height ratio
# int determines type of interpolation
# x and y have ranges of (-1, 1)
# lon is (-pi,pi) or (-pi/2, pi/2) as appropriate
# lat is (-pi/2, pi/2)
# with (0,0) at the map center


# Then, functions by projection
def Equi_coords(x, y, ex):
    lon = ma.pi * x
    lat = ma.pi * y / 2
    return lon, lat


def Equi_pos(lon, lat, ex):
    x = lon / ma.pi
    y = 2 * lat / ma.pi
    return x, y


Equi_vis = Def_vis
Equi_pres = Def_pres
Equi_typ = Def_typ
Equi_rat = 2
Equi_int = Rect_int


def Sin_coords(x, y, ex):
    lat = ma.pi * y / 2
    lon = ma.pi * x / np.cos(lat)
    return lon, lat


def Sin_pos(lon, lat, ex):
    x = np.cos(lat) * lon / ma.pi
    y = 2 * lat / ma.pi
    return x, y


def Sin_vis(x, y, lat):
    return np.where(abs(x) > np.cos(ma.pi * y / 2), False, True)


Sin_pres = Def_pres
Sin_typ = Def_typ
Sin_rat = 2
Sin_int = Def_int


def Moll_coords(x, y, ex):
    th = np.arcsin(y)
    lat = np.arcsin((2 * th + np.sin(2 * th)) / ma.pi)
    lon = np.fmod(ma.pi * x / (np.cos(th)), ma.pi)
    return lon, lat


def Moll_pos(lon, lat, ex):
    print("  (using iterative method, may take a bit)")
    t = ex[0]
    imax = ex[1]
    i = 1
    th = lat
    while (
        np.amax(np.abs(2 * th + np.sin(2 * th) / ma.pi * np.sin(lat) - 1)) > t
    ):  # no closed-form solution, so iterate until maximum error is < tolerance
        th = th - (2 * th + np.sin(2 * th) - ma.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * th)
        )
        i += 1
        if i > imax:
            print(
                "  Reached maximum of "
                + str(imax)
                + " iterations without converging, outputting result"
            )
            break
    x = lon * np.cos(th) / ma.pi
    y = np.sin(th)
    return x, y


Moll_vis = Circ_vis
Moll_pres = Def_pres
Moll_typ = Iter_typ
Moll_rat = 2
Moll_int = No_pos_int


def Hammer_coords(x, y, ex):
    x1 = x * ma.sqrt(2) * 2
    y1 = y * ma.sqrt(2)
    z = np.sqrt(1 - (x1 / 4) ** 2 - (y1 / 2) ** 2)
    lon = 2 * np.arctan(z * x1 / (2 * (2 * z**2 - 1)))
    lat = np.arcsin(z * y1)
    return lon, lat


def Hammer_pos(lon, lat, ex):
    x = np.cos(lat) * np.sin(lon / 2) / np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
    y = np.sin(lat) / np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
    return x, y


Hammer_vis = Circ_vis
Hammer_pres = Def_pres
Hammer_typ = Def_typ
Hammer_rat = 2
Hammer_int = Def_int


def Ait_guess(x, y, ex):  # Special routine for Ait_coords initial guess
    return np.where(Circ_vis(x, y, ex), Hammer_coords(x, y, ex), Wag_coords(x, y, ex))


def Ait_coords(x, y, ex):
    return Inv_coords(x, y, ex[0], ex[1], Ait_pos, Ait_guess, Ait_vis)


def Ait_pos(lon, lat, ex):
    al = ma.pi / 2 * np.sinc(np.arccos(np.cos(lat) * np.cos(lon / 2)) / ma.pi)
    x = np.cos(lat) * np.sin(lon / 2) / al
    y = np.sin(lat) / al
    return x, y


Ait_vis = Circ_vis
Ait_pres = Def_pres
Ait_typ = Iter_typ
Ait_rat = 2
Ait_int = No_coords_int


def Wink_coords(x, y, ex):
    return Inv_coords(x, y, ex[0], ex[1], Wink_pos, Wag_coords, Wink_vis)


def Wink_pos(lon, lat, ex):
    al = np.arccos(np.cos(lat) * np.cos(lon / 2)) / ma.pi
    x = (lon / ma.pi + np.cos(lat) * np.sin(lon / 2) / np.sinc(al)) / (1 + ma.pi / 2)
    y = (lat + np.sin(lat) / np.sinc(al)) / ma.pi
    return x, y


def Wink_vis(x, y, ex):
    lat1 = np.arccos(2 * (np.abs(x) * (1 + ma.pi / 2) - 1) / ma.pi)
    ymax = (lat1 + ma.pi / 2 * np.sin(lat1)) / ma.pi
    return np.where(np.abs(y) > ymax, False, True)


Wink_pres = Def_pres
Wink_typ = Iter_typ
Wink_rat = (1 + ma.pi / 2) / (ma.pi / 2)
Wink_int = No_coords_int


def Wag_coords(x, y, ex):
    lat = y * ma.pi / 2
    ph = np.arcsin(lat * ma.sqrt(3) / ma.pi)
    lon = x * ma.pi / np.cos(ph)
    return lon, lat


def Wag_pos(lon, lat, ex):
    y = lat * 2 / ma.pi
    x = lon / ma.pi * np.sqrt(1 - 3 * (lat / ma.pi) ** 2)
    return x, y


def Wag_vis(x, y, ex):
    return np.where(abs(x) > np.sqrt(1 - 3 * (y / 2) ** 2), False, True)


Wag_pres = Def_pres
Wag_typ = Def_typ
Wag_rat = 2
Wag_int = Def_int


Kav_coords = Wag_coords
Kav_pos = Wag_pos
Kav_vis = Wag_vis
Kav_pres = Wag_pres
Kav_typ = Wag_typ
Kav_rat = ma.sqrt(3)
Kav_int = Wag_int


def Ort_coords(x, y, ex):
    lon1, lat1 = Inv_coords(x, y, ex[0], ex[1], Ort_pos, Hammer_coords, Circ_vis)
    lat = y * ma.pi / 2
    lon2 = np.abs(x) * ma.pi + ma.pi / 2 - np.sqrt(ma.pi**2 / 4 - lat**2)
    lon2 = np.where(x > 0, lon2, -lon2)
    lon = np.where(np.sqrt((x * 2) ** 2 + y**2) > 1, lon2, lon1)
    return lon, lat


def Ort_pos(lon, lat, ex):
    y = lat * 2 / ma.pi
    abslon = np.abs(lon)
    F = (ma.pi**2 / (4 * abslon) + abslon) / 2
    x = np.where(
        abslon > ma.pi / 2,
        np.sqrt(ma.pi**2 / 4 - lat**2) + abslon - ma.pi / 2,
        abslon - F + np.sqrt(F**2 - (y * ma.pi / 2) ** 2),
    )
    x = np.where(lon > 0, x / ma.pi, -x / ma.pi)
    return x, y


def Ort_vis(x, y, ex):
    return np.where(np.abs(x) > 1 / 2, Hem_vis(x, y, 2), True)


Ort_pres = Def_pres
Ort_typ = Iter_typ
Ort_rat = 2
Ort_int = No_coords_int


def Nic_coords1(x, y, ex):
    return Inv_coords(x, y, ex[0], ex[1], Nic_pos, Azim_coords, Nic_vis, ex, ex[2])


def Nic_pos1(lon, lat, ex):
    R = 2 / ma.pi
    lat0 = np.where(lat == 0, True, False)
    lon0 = np.where(lon == 0, True, False)
    latmax = np.where(np.abs(lat) == ma.pi / 2, True, False)
    lonmax = np.where(np.abs(lon) == ma.pi / 2, True, False)
    lat1 = np.where(lat0 | latmax, lat + 1e-8, lat)
    lon1 = np.where(lon0 | lonmax, lon + 1e-8, lon)
    sinla = np.sin(lat1)
    b = ma.pi / (2 * lon1) - 2 * lon1 / ma.pi
    c = 2 * lat1 / ma.pi
    d = (1 - c**2) / (sinla - c)
    b2 = b**2
    d2 = d**2
    b2d2 = 1 + b2 / d2
    d2b2 = 1 + d2 / b2
    M = (b * sinla / d - b / 2) / b2d2
    N = (d2 * sinla / b2 + d / 2) / d2b2
    x1 = np.sqrt(M**2 + np.cos(lat1) ** 2 / b2d2)
    x = np.where(lon > 0, M + x1, M - x1) * R * ma.pi / 2
    y1 = np.sqrt(N**2 - (d2 / b2 * sinla**2 + d * sinla - 1) / d2b2)
    y = np.where(lat < 0, N + y1, N - y1) * R * ma.pi / 2
    x = np.where(lon0 | latmax, 0, x)
    x = np.where(lat0 | lonmax, np.cos(lat) * lon * R, x)
    y = np.where(lon0 | lat0, R * lat, y)
    y = np.where(lonmax, np.sin(lat) * R * ma.pi / 2, y)
    y = np.where(latmax, R * lat, y)
    x = np.where(lon > ma.pi / 2, ma.pi - x, np.where(lon < -ma.pi / 2, -ma.pi - x, x))
    return x, y


def Nic_coords(x, y, ex):
    if ex[2] == 0:
        return Nic_coords1(x, y, ex)
    elif ex[2] == 1:
        return Onehem_coords(x, y, ex, Nic_coords1, 1)
    elif (
        ex[2] == 2
    ):  # Iteration doesn't like to work for bihemispheres, so the arrays are split, each half projected separately, and then rejoined
        x1, x2 = np.array_split(x, 2, 1)
        y1, y2 = np.array_split(y, 2, 1)
        lon1, lat1 = Onehem_coords(2 * x1 + 1, y1, (ex[0], ex[1], 1), Nic_coords1, 1)
        lon2, lat2 = Onehem_coords(2 * x2 - 1, y2, (ex[0], ex[1], 1), Nic_coords1, 1)
        lon = np.concatenate((lon1, lon2), 1)
        lat = np.concatenate((lat1, lat2), 1)
        lon = np.where(x > 0, lon + ma.pi / 2, lon - ma.pi / 2)
        return lon, lat
    else:
        raise Exception("Invalid output map subtype selection (must be 0, 1, or 2)")


def Nic_pos(lon, lat, ex):
    return Hem_pos(lon, lat, ex[2], ex, Nic_pos1, 1)


def Nic_vis(x, y, ex):
    return Hem_vis(x, y, ex[2])


def Nic_pres(x, y, ex):
    return Hem_pres(x, y, ex[2])


Nic_typ = HemonlyIter_typ
Nic_rat = 1
Nic_int = No_coords_int


def Eckiv_coords(x, y, ex):
    r = 1 / (2 * ma.sqrt(ma.pi / (4 + ma.pi)))
    th = np.arcsin(y * np.sqrt(4 + ma.pi) / (2 * np.sqrt(ma.pi) * r))
    lat = np.arcsin((th + np.sin(th) * np.cos(th) + 2 * np.sin(th)) / (2 + ma.pi / 2))
    lon = 2 * x * np.sqrt(4 * ma.pi + ma.pi**2) / (2 * r * (1 + np.cos(th)))
    return lon, lat


def Eckiv_pos(lon, lat, ex):
    print("  (using iterative method, may take a bit)")
    t = ex[0]
    imax = ex[1]
    i = 1
    th = lat / 2
    while (
        np.amax(
            np.abs(
                (th + np.sin(th) * np.cos(th) + 2 * np.sin(th))
                - ((2 + ma.pi / 2) * np.sin(lat))
            )
        )
        > t
    ):  # no closed-form solution, so iterate until maximum error is < tolerance
        th = th - (
            th
            + np.sin(th) * np.cos(th)
            + 2 * np.sin(th)
            - (2 + ma.pi / 2) * np.sin(lat)
        ) / (2 * np.cos(th) * (1 + np.cos(th)))
        i += 1
        if i > imax:
            print(
                "  Reached maximum of "
                + str(imax)
                + " iterations without converging, outputting result"
            )
            break
    r = 1
    x = (
        lon * (1 + np.cos(th)) / (2 * ma.pi)
    )  # (1 / np.sqrt(4*ma.pi + ma.pi**2)) * lon * (1 + np.cos(th)) *r
    y = np.sin(th)  # 2 * np.sqrt(ma.pi / (4 + ma.pi)) * np.sin(th) *r
    return x, y


Eckiv_vis = Ort_vis
Eckiv_pres = Def_pres
Eckiv_typ = Iter_typ
Eckiv_rat = 2
Eckiv_int = No_pos_int


def Azim_coords1(x, y, ex):
    rh = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, -y)
    return From_pol(rh, th)


def Azim_pos1(lon, lat, ex):
    rh, th = To_pol(lon, lat)
    x = rh * np.sin(th)
    y = rh * np.cos(th)
    return x, y


def Azim_coords(x, y, hem):
    return Hem_coords(x, y, hem, 0, Azim_coords1, 1 / 2)


def Azim_pos(x, y, hem):
    return Hem_pos(x, y, hem, 0, Azim_pos1, 1 / 2)


Azim_vis = Hem_vis
Azim_pres = Hem_pres
Azim_typ = Hemfull_typ
Azim_rat = 1
Azim_int = Def_int


def Ortho_coords1(x, y, ex):
    rh = np.sqrt((2 * x) ** 2 + (2 * y) ** 2)
    c = np.arcsin(np.fmod(rh, 1))
    lat = np.arcsin(2 * y * np.sin(c) / rh)
    lon = np.arctan(2 * x * np.sin(c) / (rh * np.cos(c)))
    return lon, lat


def Ortho_pos1(lon, lat, ex):
    x = np.cos(lat) * np.sin(lon) / 2
    y = np.sin(lat) / 2
    return x, y


def Ortho_coords(x, y, hem):
    return Hem_coords(x, y, hem, 0, Ortho_coords1, 1 / 2)


def Ortho_pos(x, y, hem):
    return Hem_pos(x, y, hem, 0, Ortho_pos1, 1 / 2)


Ortho_vis = Hem_vis
Ortho_pres = Hem_pres
Ortho_typ = Hemonly_typ
Ortho_rat = 1
Ortho_int = Def_int


def Stereo_coords1(x, y, ex):
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, -y)
    rh = (np.arctan(2 * r) - ma.pi / 4) * 2 / ma.pi + 1 / 2
    return From_pol(rh, th)


def Stereo_pos1(lon, lat, ex):
    rh, th = To_pol(lon, lat)
    r = np.tan(ma.pi / 4 + ma.pi * (rh - 1 / 2) / 2) / 2
    x = r * np.sin(th)
    y = r * np.cos(th)
    return x, y


def Stereo_coords(x, y, hem):
    return Hem_coords(x, y, hem, 0, Stereo_coords1, 1 / 2)


def Stereo_pos(x, y, hem):
    return Hem_pos(x, y, hem, 0, Stereo_pos1, 1 / 2)


Stereo_vis = Hem_vis
Stereo_pres = Hem_pres
Stereo_typ = Hemonly_typ
Stereo_rat = 1
Stereo_int = Def_int


def Lamb_coords1(x, y, ex):
    rh = np.sqrt(x**2 + y**2)
    rh = np.fmod(rh, 1)
    th = np.arctan2(x, -y)
    rh1 = 2 * np.arcsin(rh) / ma.pi
    return From_pol(rh1, th)


def Lamb_pos1(lon, lat, ex):
    rh, th = To_pol(lon, lat)
    rh1 = np.sin(rh * ma.pi / 2)
    x = rh1 * np.sin(th)
    y = rh1 * np.cos(th)
    return x, y


def Lamb_coords(x, y, hem):
    return Hem_coords(x, y, hem, 0, Lamb_coords1, ma.sqrt(2) / 2)


def Lamb_pos(x, y, hem):
    return Hem_pos(x, y, hem, 0, Lamb_pos1, ma.sqrt(2) / 2)


Lamb_vis = Hem_vis
Lamb_pres = Hem_pres
Lamb_typ = Hemfull_typ
Lamb_rat = 1
Lamb_int = Def_int


def Merc_coords(x, y, ex):
    lon = ma.pi * x
    lat = y / abs(y) * (2 * np.arctan(np.exp(abs(y) * ma.pi)) - ma.pi / 2)
    return lon, lat


def Merc_pos(lon, lat, ex):
    x = lon / ma.pi
    y = lat / abs(lat) * np.log(np.tan(ma.pi / 4 + abs(lat) / 2)) / ma.pi
    return x, y


Merc_vis = Def_vis


def Merc_pres(x, y, ex):
    return np.where(abs(y) > 1, False, True)


Merc_typ = Def_typ
Merc_rat = 1
Merc_int = Rect_int


def Gallst_coords(x, y, ex):
    lon = x * ma.pi
    lat = 2 * np.arctan(y)
    return lon, lat


def Gallst_pos(lon, lat, ex):
    x = lon / ma.pi
    y = np.tan(lat / 2)
    return x, y


Gallst_vis = Def_vis
Gallst_pres = Def_pres
Gallst_typ = Def_typ
Gallst_rat = (ma.pi / ma.sqrt(2)) / (1 + ma.sqrt(2) / 2)
Gallst_int = Rect_int


def Mill_coords(x, y, ex):
    lon = x * ma.pi
    y1 = y * ma.asinh(ma.tan(ma.pi * 2 / 5))
    lat = 5 / 4 * np.arctan(np.sinh(y1))
    return lon, lat


def Mill_pos(lon, lat, ex):
    x = lon / ma.pi
    y1 = np.arcsinh(np.tan(lat * 4 / 5))
    y = y1 / ma.asinh(ma.tan(ma.pi * 2 / 5))
    return x, y


Mill_vis = Def_vis
Mill_pres = Def_pres
Mill_typ = Def_typ
Mill_rat = ma.pi / (5 / 4 * ma.asinh(ma.tan(ma.pi * 2 / 5)))
Mill_int = Rect_int

################################# Equatorial Collignon Quincuncial Projection

# Remapping functions for each projection:
# coords determines (lon,lat) coordinates for a given (x,y) position on the map
# pos determines (x,y) position for given (lon,lat) coordinates
# vis determines if an (x,y) position on the output should be shown on non-rectangular maps
# pres determines if an (x,y) position is present on the input
# typ incorporates extra information required for specific projection types
# rat is the map width/height ratio
# int determines type of interpolation
# x and y have ranges of (-1, 1)
# lon is (-pi,pi) or (-pi/2, pi/2) as appropriate
# lat is (-pi/2, pi/2)
# with (0,0) at the map center


def Coll_coords(x, y, ex):
    long, lat = cq.collignon_quincuncial_inverse(x, y)
    return long, lat


def Coll_pos(phi, lambda_, ex):
    x_t, y_t = cq.collignon_quincuncial(phi, lambda_)
    return x_t, y_t


Coll_vis = Def_vis
Coll_pres = Def_pres
Coll_rat = 1
Coll_typ = Def_typ
Coll_int = Def_int


def Petrov_coords(x, y, ex):
    lon, lat = pq.new_projection_inverse(x, y)
    return lon, lat


def Petrov_pos(lon, lat, ex):
    x, y = pq.new_projection(lon - np.pi, lat)
    return x, y


Petrov_vis = Def_vis
Petrov_pres = Def_pres
Petrov_rat = 1
Petrov_typ = Def_typ
Petrov_int = Def_int

# List of all currently implemented projections; finding their index in this list will give you the proper index to use for running the functions
proj_list = [
    "Equirectangular",
    "Sinusoidal",
    "Mollweide",
    "Hammer",
    "Aitoff",
    "Winkel Tripel",
    "Kavrayskiy VII",
    "Wagner VI",
    "Ortelius Oval",
    "Nicolosi Globular",
    "Eckert IV",
    "Azimuthal Equidistant",
    "Orthographic",
    "Stereographic",
    "Lambert Azimuthal",
    "Mercator",
    "Gall Stereographic",
    "Miller Cylindrical",
    "Equatorial Collignon QuincuncialPetrov Equal-Area Quincuncial",
]


# Lists referring to the functions in same order for easy reference
# Nicolosi and Lambert still buggy
coordsl = [
    Equi_coords,
    Sin_coords,
    Moll_coords,
    Hammer_coords,
    Ait_coords,
    Wink_coords,
    Kav_coords,
    Wag_coords,
    Ort_coords,
    Nic_coords,
    Eckiv_coords,
    Azim_coords,
    Ortho_coords,
    Stereo_coords,
    Lamb_coords,
    Merc_coords,
    Gallst_coords,
    Mill_coords,
    Coll_coords,
    Petrov_coords,
]

posl = [
    Equi_pos,
    Sin_pos,
    Moll_pos,
    Hammer_pos,
    Ait_pos,
    Wink_pos,
    Kav_pos,
    Wag_pos,
    Ort_pos,
    Nic_pos,
    Eckiv_pos,
    Azim_pos,
    Ortho_pos,
    Stereo_pos,
    Lamb_pos,
    Merc_pos,
    Gallst_pos,
    Mill_pos,
    Coll_pos,
    Petrov_pos,
]

visl = [
    Equi_vis,
    Sin_vis,
    Moll_vis,
    Hammer_vis,
    Ait_vis,
    Wink_vis,
    Kav_vis,
    Wag_vis,
    Ort_vis,
    Nic_vis,
    Eckiv_vis,
    Azim_vis,
    Ortho_vis,
    Stereo_vis,
    Lamb_vis,
    Merc_vis,
    Gallst_vis,
    Mill_vis,
    Coll_vis,
    Petrov_vis,
]

presl = [
    Equi_pres,
    Sin_pres,
    Moll_pres,
    Hammer_pres,
    Ait_pres,
    Wink_pres,
    Kav_pres,
    Wag_pres,
    Ort_pres,
    Nic_pres,
    Eckiv_pres,
    Azim_pres,
    Ortho_pres,
    Stereo_pres,
    Lamb_pres,
    Merc_pres,
    Gallst_pres,
    Mill_pres,
    Coll_pres,
    Petrov_pres,
]

ratl = [
    Equi_rat,
    Sin_rat,
    Moll_rat,
    Hammer_rat,
    Ait_rat,
    Wink_rat,
    Kav_rat,
    Wag_rat,
    Ort_rat,
    Nic_rat,
    Eckiv_rat,
    Azim_rat,
    Ortho_rat,
    Stereo_rat,
    Lamb_rat,
    Merc_rat,
    Gallst_rat,
    Mill_rat,
    Coll_rat,
    Petrov_rat,
]

typl = [
    Equi_typ,
    Sin_typ,
    Moll_typ,
    Hammer_typ,
    Ait_typ,
    Wink_typ,
    Kav_typ,
    Wag_typ,
    Ort_typ,
    Nic_typ,
    Eckiv_typ,
    Azim_typ,
    Ortho_typ,
    Stereo_typ,
    Lamb_typ,
    Merc_typ,
    Gallst_typ,
    Mill_typ,
    Coll_typ,
    Petrov_typ,
]

intl = [
    Equi_int,
    Sin_int,
    Moll_int,
    Hammer_int,
    Ait_int,
    Wink_int,
    Kav_int,
    Wag_int,
    Ort_int,
    Nic_int,
    Eckiv_int,
    Azim_int,
    Ortho_int,
    Stereo_int,
    Lamb_int,
    Merc_int,
    Gallst_int,
    Mill_int,
    Coll_int,
    Petrov_int,
]


# Runs the main function but directly takes angle inputs in degrees
def MainDeg(
    file_in,
    file_out,
    proj_in=0,
    proj_out=0,
    lon_in=0.0,
    lat_in=0.0,
    rot_in=0.0,
    lon_out=0.0,
    lat_out=0.0,
    rot_out=0.0,
    tol=1e-6,
    imax=20,
    hem_in=0,
    hem_out=0,
    trim=0,
    trunc=False,
    interp=0,
    aviter=False,
):
    Main(
        file_in,
        file_out,
        proj_in,
        proj_out,
        ma.radians(lon_in),
        ma.radians(lat_in),
        ma.radians(rot_in),
        ma.radians(lon_out),
        ma.radians(lat_out),
        ma.radians(rot_out),
        tol,
        imax,
        hem_in,
        hem_out,
        trim,
        trunc,
        interp,
        aviter,
    )


# Input


def Inprompt(prompt, f, proj=False):
    inp = input(prompt)
    while True:
        try:
            var = f(inp)
            if proj:
                a = typl[var]
            return var
        except:
            inp = input(" Invalid input, try again: ")


if __name__ == "__main__":
    hem_in = 0
    hem_out = 0
    print("""
Projection Pasta
For Reprojection of maps between arbitrary aspects
Made 2023 by Amadea de Silva and Nikolai Hersfeldt
Forked 2025 by Jessica Butterfield

Projection Options and Codes (with profile):
  0: Equirectangular/Plate Caree (2:1 rectangle)
  1: Sinusoidal (2:1 sinusoid)
  2: Mollweide (2:1 ellipse)
  3: Hammer (2:1 ellipse)
  4: Aitoff (2:1 ellipse)
  5: Winkel Tripel (1.637:1 ovalish)
  6: Kavrayskiy VII (1.732:1 ovalish)
  7: Wagner VI (2:1 ovalish)
  8: Ortelius Oval (2:1 oval)
  9: Nicolosi Globular (1:1 hemisphere)
 10: Eckert IV (2:1 oval)
 11: Azimuthal Equidistant (1:1 circle)
 12: Orthographic (1:1 hemisphere)
 13: Stereographic (1:1 hemisphere)
 14: Lambert Azimuthal Equal-Area (1:1 circle)
 15: Mercator truncated to square (1:1 square)
 16: Gall Stereographic (1.301:1 rectangle)
 17: Miller Cylindrical (1.364:1 rectangle)
 18: Collignon Quincuncial (1:1 square)
 19: Petrov Equal-Area Quincuncial (1:1 square)

""")
    print("Input Image")
    while True:
        file_in = input(" Filename: ")
        if os.path.exists(file_in):
            break
        print("  No file found at " + str(file_in))
    proj_in = Inprompt(" Projection: ", int, True)
    if typl[proj_in] == Hemfull_typ:
        hem_in = Inprompt("  0 for global, 1 for hemisphere, 2 for bihemisphere: ", int)
    elif typl[proj_in] == Hemonly_typ or typl[proj_in] == HemonlyIter_typ:
        hem_in = Inprompt("  1 for hemisphere, 2 for bihemisphere: ", int)
    lon_in = ma.radians(Inprompt(" Center longitude (-180 to 180): ", float))
    lat_in = ma.radians(Inprompt(" Center latitude (-90 to 90): ", float))
    rot_in = ma.radians(Inprompt(" Clockwise rotation from north (0 to 360): ", float))

    print("""
Output Image""")
    file_out = input(" Filename: ")
    proj_out = Inprompt(" Projection: ", int, True)
    if typl[proj_out] == Hemfull_typ:
        hem_out = Inprompt(
            "  0 for global, 1 for hemisphere, 2 for bihemisphere: ", int
        )
    elif typl[proj_out] == Hemonly_typ or typl[proj_out] == HemonlyIter_typ:
        hem_out = Inprompt("  1 for hemisphere, 2 for bihemisphere: ", int)
    lon_out = ma.radians(Inprompt(" Center longitude (-180 to 180): ", float))
    lat_out = ma.radians(Inprompt(" Center latitude (-90 to 90): ", float))
    rot_out = ma.radians(Inprompt(" Clockwise rotation from north (0 to 360): ", float))
    print("""
Interpolation:
 0: None; may have artifacts in some projections, but requires no scipy install
 1: Nearest; copies nearest pixel
 2: Linear (blurry but most reliable)
 3: Cubic (very slow)
 4: PCHIP (even slower)
 5: Spline""")
    interp = Inprompt("Interpolation type: ", int)
    trim = input("""
Trim map edges?
 Specify pixel coordinates on output map to crop output to
 Only these pixels will be projected, so saves time
 (but be sure to expand to full map dimensions if you want to return to project again)
 y/n: """)
    if "y" in trim or "Y" in trim or "1" in trim:
        trim = [0, 0, 0, 0]
        trim[0] = Inprompt("   Left edge pixel index: ", int)
        trim[1] = Inprompt("    Top edge pixel index: ", int)
        trim[2] = Inprompt("  Right edge pixel index: ", int)
        trim[3] = Inprompt(" Bottom edge pixel index: ", int)
    else:
        trim = 0
    trunc = input("""
Limit map output to single globe surface?
 Limited maps may have missing pixels on edges when reprojected a second time
 Unlimited maps may have odd noise on edges in some projections (outside of the usually cropped area)
 y/n: """)
    if "y" in trunc or "Y" in trunc or "1" in trunc:
        trunc = True
    else:
        trunc = False

    print("""
Working...""")

    Main(
        file_in,
        file_out,
        proj_in,
        proj_out,
        lon_in,
        lat_in,
        rot_in,
        lon_out,
        lat_out,
        rot_out,
        hem_in=hem_in,
        hem_out=hem_out,
        trim=trim,
        trunc=trunc,
        interp=interp,
    )

    z = input("Press enter to close")
