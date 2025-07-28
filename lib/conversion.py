# Finds the corresponding pixel index in input for every index in the output

from PIL import Image
import numpy as np
import math as ma
from lib.geometry import Rotate_from, Rotate_to
from lib.projection_utilities import Def_vis, Rect_int
from main import coordsl, intl, posl, presl, proj_list, ratl, typl, visl


def Find_index(
    x1,
    y1,
    coords,
    pos,
    lon_in,
    lat_in,
    rot_in,
    ex_in,
    lon_out,
    lat_out,
    rot_out,
    ex_out,
):
    print(" Determining lat/lon of output pixels...")
    lon1, lat1 = coords(x1, y1, ex_out)
    print(" Rotating from output orientation...")
    lon1, lat1 = Rotate_from(lon1, lat1, lon_out, lat_out, rot_out)
    print(" Rotating to input orientation...")
    lon1, lat1 = Rotate_to(lon1, lat1, lon_in, lat_in, rot_in)
    print(" Determining corresponding input pixels...")
    x2, y2 = pos(lon1, lat1, ex_in)
    return y2, x2


def Find_data(
    data_in,
    x_in,
    y_in,
    x_out,
    y_out,
    interp,
    proj_in,
    proj_out,
    lon_in,
    lat_in,
    rot_in,
    ex_in,
    lon_out,
    lat_out,
    rot_out,
    ex_out,
):
    if interp == 5:
        if intl[proj_in] == Rect_int:
            from scipy.interpolate import RectSphereBivariateSpline as scint
        else:
            from scipy.interpolate import RectBivariateSpline as scint
    elif interp > 0:
        from scipy.interpolate import RegularGridInterpolator as scint
    print(" Determining lat/lon of output pixels...")
    lon1, lat1 = coordsl[proj_out](x_out, y_out, ex_out)
    print(" Rotating from output orientation...")
    lon1, lat1 = Rotate_from(lon1, lat1, lon_out, lat_out, rot_out)
    print(" Rotating to input orientation...")
    lon1, lat1 = Rotate_to(lon1, lat1, lon_in, lat_in, rot_in)
    if interp == 5:
        if intl[proj_in] == Rect_int:
            print(" Determining lat/lon of input pixels...")
            x_in, y_in = np.meshgrid(x_in, y_in)
            lon2, lat2 = coordsl[proj_in](x_in, y_in, ex_in)
            in1 = np.ravel(ma.pi / 2 - lat2[:, 0])
            in2 = np.ravel(ma.pi + lon2[0, :])
            out1 = np.ravel(np.remainder(ma.pi / 2 - lat1, ma.pi))
            out2 = np.ravel(np.remainder(ma.pi + lon1, 2 * ma.pi))
            if proj_in == proj_list.index("Mercator"):
                x1, y1 = posl[proj_in](lon1, lat1, ex_in)
            else:
                x1 = 0
                y1 = 0
        else:
            print(" Determining corresponding input pixels...")
            x1, y1 = posl[proj_in](lon1, lat1, ex_in)
            x1 = np.where(
                np.abs(x1) > 1, (np.abs(x1 - 1) % 2 - 1) * np.where(x1 > 0, 1, -1), x1
            )  # x wraps around
            y1 = np.where(
                np.abs(y1) > 1, np.abs(np.abs(y1 - 1) % 4 - 2) - 1, y1
            )  # y inverts
            out1 = np.ravel(0 - y1)
            out2 = np.ravel(x1)
            in1 = 0 - y_in
            in2 = x_in
        print(" Interpolating input data to output pixels...")
        if data_in.ndim > 2:
            data_out = np.zeros(
                (x_out.shape[0], x_out.shape[1], data_in.shape[2]), data_in.dtype
            )
            for n in range(data_in.shape[2]):
                spline = scint(in1, in2, data_in[:, :, n])
                data_out[:, :, n] = (
                    spline.ev(out1, out2).reshape((x_out.shape)).astype(data_in.dtype)
                )
        else:
            spline = scint(in1, in2, data_in)
            data_out = (
                spline.ev(out1, out2).reshape((x_out.shape)).astype(data_in.dtype)
            )
    else:
        types = ["none", "nearest", "linear", "cubic", "pchip"]
        print(" Determining corresponding input pixels...")
        x1, y1 = posl[proj_in](lon1, lat1, ex_in)
        x1 = np.where(
            np.abs(x1) > 1, (np.abs(x1 - 1) % 2 - 1) * np.where(x1 > 0, 1, -1), x1
        )  # x wraps around
        y1 = np.where(
            np.abs(y1) > 1, np.abs(np.abs(y1 - 1) % 4 - 2) - 1, y1
        )  # y inverts
        print(" Interpolating input data to output pixels...")
        if data_in.ndim > 2:
            data_out = np.zeros(
                (x_out.shape[0], x_out.shape[1], data_in.shape[2]), data_in.dtype
            )
            for n in range(data_in.shape[2]):
                inter = scint(
                    (y_in, x_in),
                    data_in[:, :, n].astype(float),
                    method=types[interp],
                    bounds_error=False,
                    fill_value=None,
                )
                data_out[:, :, n] = inter(np.stack((y1, x1), -1)).astype(data_in.dtype)
        else:
            inter = scint(
                (y_in, x_in),
                data_in.astype(float),
                method=types[interp],
                bounds_error=False,
                fill_value=None,
            )
            data_out = inter(np.stack((y1, x1), -1)).astype(data_in.dtype)
    return data_out, x1, y1


# Main function


def Main(
    file_in,
    file_out,
    proj_in=0,
    proj_out=0,
    lon_in=0,
    lat_in=0,
    rot_in=0,
    lon_out=0,
    lat_out=0,
    rot_out=0,
    tol=1e-6,
    imax=20,
    hem_in=0,
    hem_out=0,
    trim=0,
    trunc=False,
    interp=0,
    aviter=False,
):
    if trunc:
        vis = visl[proj_out]
    else:
        vis = Def_vis
    pres = presl[proj_in]
    ex_in = typl[proj_in](tol, imax, hem_in)
    ex_out = typl[proj_out](tol, imax, hem_out)
    rat = ratl[proj_in] / ratl[proj_out]
    if hem_in == 2:
        rat *= 2
    if hem_out == 2:
        rat /= 2
    map_in = Image.open(file_in)
    mapw = map_in.width
    maph = map_in.height
    mapw_out = mapw
    maph_out = maph
    if rat > 1:
        maph_out = int(
            maph * rat
        )  # increase rather than decrease resolution as necessary to reach new aspect ratio
    elif rat < 1:
        mapw_out = int(mapw / rat)
    tol = min(tol, 0.1 / mapw, 0.1 / maph)

    data_in = np.asarray(map_in)

    # if data_in.ndim > 2:
    #    blank = np.zeros(data_in[0,0].shape)    #Create blank color appropriate to image color mode
    # else:
    #    blank = 0
    x_out = np.linspace(
        -1 + 1 / mapw_out, 1 - 1 / mapw_out, mapw_out
    )  # All maps are treated internally as squares
    y_out = np.linspace(1 - 1 / maph_out, -1 + 1 / maph_out, maph_out)
    if trim != 0:
        x_out = x_out[
            trim[0] : trim[2]
        ]  # trim at start, so all future steps are applied only for desired pixels
        y_out = y_out[trim[1] : trim[3]]
    x_out, y_out = np.meshgrid(x_out, y_out)
    if interp == 0:
        coords = coordsl[proj_out]
        pos = posl[proj_in]
        y_in, x_in = Find_index(
            x_out,
            y_out,
            coords,
            pos,
            lon_in,
            lat_in,
            rot_in,
            ex_in,
            lon_out,
            lat_out,
            rot_out,
            ex_out,
        )
        x_in2 = np.where(
            np.abs(x_in) > 1,
            (np.abs(x_in - 1) % 2 - 1) * np.where(x_in > 0, 1, -1),
            x_in,
        )  # x wraps around
        y_in2 = np.where(
            np.abs(y_in) > 1, np.abs(np.abs(y_in - 1) % 4 - 2) - 1, y_in
        )  # y inverts
        data_out = data_in[
            (
                np.rint((1 - y_in2) * maph / 2 - 0.5).astype(np.int64),
                np.rint((x_in2 + 1) * mapw / 2 - 0.5).astype(np.int64),
            )
        ]
    else:
        x_in = np.linspace(-1 + 1 / mapw, 1 - 1 / mapw, mapw)
        y_in = np.linspace(1 - 1 / maph, -1 + 1 / maph, maph)
        data_out, x_in2, y_in2 = Find_data(
            data_in,
            x_in,
            y_in,
            x_out,
            y_out,
            interp,
            proj_in,
            proj_out,
            lon_in,
            lat_in,
            rot_in,
            ex_in,
            lon_out,
            lat_out,
            rot_out,
            ex_out,
        )
    print(" Remapping pixels...")
    # print(np.amax(x_in))
    # print(np.amax(y_in))
    vis1 = vis(x_out, y_out, ex_out)
    if interp == 5 and proj_in != proj_list.index("Mercator"):
        pres1 = True
    else:
        pres1 = pres(x_in2, y_in2, ex_in)
    if (
        data_in.ndim > 2
    ):  # various procedures to allow the visible and present masks to be cast out to the appropriate array shape
        vis2 = np.expand_dims(vis1, -1)
        vis3 = vis2
        pres2 = np.expand_dims(pres1, -1)
        pres3 = pres2
        if data_in.shape[2] > 1:
            for i in range(data_in.shape[2] - 1):
                vis3 = np.concatenate((vis3, vis2), -1)
                pres3 = np.concatenate((pres3, pres2), -1)
    else:
        vis3 = vis1
        pres3 = pres1
    blank = np.zeros_like(data_out)
    data_out = np.where(vis3 & pres3, data_out, blank).astype(data_in.dtype)
    print(" Outputting image...")
    map_out = Image.fromarray(data_out, mode=map_in.mode)
    if map_in.mode == "P":
        map_out.putpalette(map_in.getpalette())
    map_out.save(file_out)
    print("Finished; saved to " + file_out)
