# First, some generic functions used for multiple projections
import numpy as np
import math as ma


def From_pol(rh, th):  # Converts from polar around equator to lon,lat
    rh *= ma.pi
    lat = np.arcsin(-np.sin(rh) * np.cos(th))
    lon = np.arctan2(np.sin(th) * np.sin(rh), np.cos(rh))
    return lon, lat


def To_pol(lon, lat):  # Converts from lon,lat to polar around equator
    rh = np.arccos(np.cos(lat) * np.cos(lon)) / ma.pi
    th = np.arctan2(np.sin(lon) * np.cos(lat), np.sin(lat))
    return rh, th


def der(lon, lat, t, f, ex):  # Approximate partial derivatives by the secant method
    x1, y1 = f(lon, lat + t, ex)
    x2, y2 = f(lon, lat - t, ex)
    x3, y3 = f(lon + t, lat, ex)
    x4, y4 = f(lon - t, lat, ex)
    dxla = (x1 - x2) / (2 * t)
    dyla = (y1 - y2) / (2 * t)
    dxlo = (x3 - x4) / (2 * t)
    dylo = (y3 - y4) / (2 * t)
    return dxla, dxlo, dyla, dylo


def Inv_coords(
    x, y, t, imax, f, f1, vis, ex=-1, ex1=-1
):  # General iterative inverse function from Bildirici 2016: https://doi.org/10.1080/15230406.2016.1200492
    print("  (using iterative method, may take a bit)")
    if ex == -1:
        ex = t
    if ex1 == -1:
        ex1 = ex
    lon, lat = f1(x, y, ex1)
    vis1 = vis(x, y, ex)
    i = 1
    while True:
        x1, y1 = f(lon, lat, ex)
        dxla, dxlo, dyla, dylo = der(lon, lat, t, f, ex)
        div = dxla * dylo - dyla * dxlo
        divz = np.where(div != 0, True, False)
        div1 = np.where(divz, div, 1)
        difx = x1 - x
        dify = y1 - y
        dlon = np.where(divz, (dify * dxla - difx * dyla) / div1, 0)
        dlat = np.where(divz, (difx * dylo - dify * dxlo) / div1, 0)
        if (
            np.amax(np.where(vis1, np.abs(dlon), 0)) < t
        ):  # only check within the region that'll typically be visible, there are some glitchy regions outside I haven't managed to eliminate
            if np.amax(np.where(vis1, np.abs(dlat), 0)) < t:
                break
        lon = lon - dlon
        lat = lat - dlat
        i += 1
        if i > imax:
            print(
                "  Reached maximum of "
                + str(imax)
                + " iterations without converging, outputting result"
            )
            break
    return lon, lat


def Onehem_coords(
    x, y, ex, f, s
):  # Set of general functions for maps that can be mapped as hemispheres
    return f(x * s, y * s, ex)


def Bihem_coords(x, y, ex, f, s):
    lon, lat = Onehem_coords(np.where(x > 0, 2 * x - 1, 2 * x + 1), y, ex, f, s)
    lon = np.where(x > 0, lon + ma.pi / 2, lon - ma.pi / 2)
    return lon, lat


def HemonlyIter_typ(tol, imax, hem):
    return (tol, imax, hem)


def Iter_typ(tol, imax, hem):
    return (tol, imax)


def Hemonly_typ(tol, imax, hem):
    if hem == 0:
        hem = 1
    return hem


def Hemfull_typ(tol, imax, hem):
    return hem


def Def_typ(tol, imax, hem):
    return 0


def Def_vis(x, y, ex):
    return True


def Hem_pres(x, y, hem):
    if hem == 0 or hem == 2:
        return True
    elif hem == 1:
        return np.where(np.sqrt(x**2 + y**2) > 1, False, True)


def Circ_vis(x, y, ex):
    return np.where(np.sqrt(x**2 + y**2) > 1, False, True)


def Hem_vis(x, y, hem):
    if hem == 0 or hem == 1:
        return Circ_vis(x, y, 0)
    elif hem == 2:
        return Circ_vis(1 - abs(x * 2), y, 0)


def Onehem_pos(lon, lat, ex, f, s):
    x, y = f(lon, lat, ex)
    return x / s, y / s


def Bihem_pos(lon, lat, ex, f, s):
    lon1 = np.where(lon > 0, lon, lon + ma.pi) - ma.pi / 2
    x, y = Onehem_pos(lon1, lat, ex, f, s)
    x /= 2
    return np.where(lon > 0, x + 0.5, x - 0.5), y


def Hem_pos(lon, lat, hem, ex, f, s):
    if hem == 0:
        return f(lon, lat, ex)
    elif hem == 1:
        return Onehem_pos(lon, lat, ex, f, s)
    elif hem == 2:
        return Bihem_pos(lon, lat, ex, f, s)
    else:
        raise Exception("Invalid input map subtype selection (must be 0, 1, or 2)")


def Edge_vis(x, lat, f):
    xmax, y1 = f(ma.pi, np.where(np.abs(lat) > ma.pi / 2, ma.pi / 2, lat), x)
    return np.where(np.abs(x) > xmax, False, True)


def Def_pres(x, y, ex):
    return True


def Hem_coords(x, y, hem, ex, f, s):
    if hem == 0:
        return f(x, y, ex)
    elif hem == 1:
        return Onehem_coords(x, y, ex, f, s)
    elif hem == 2:
        return Bihem_coords(x, y, ex, f, s)
    else:
        raise Exception("Invalid output map subtype selection (must be 0, 1, or 2)")


Def_int = 100
Rect_int = 101
No_coords_int = 102
No_pos_int = 103
