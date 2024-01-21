#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:07:04 2019

@author: Yongzhen Fan
"""

import calendar
import time
from glob import glob
from os import listdir
from os.path import basename, exists

import h5py

# import l8angles
import numpy as np
import pyproj
from glymur import Jp2k
from lxml import objectify
from osgeo import gdal
from pyhdf.SD import SD, SDC
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt

from src.goci_utils import (
    goci_sensor_azimuth,
    goci_sensor_zenith,
    goci_slots,
    goci_slots_time,
    goci_solar_angles,
)
from src.gsw import GSW
from src.l8_utils import load_mtl

# from datetime import datetime, timedelta


class L1B(object):
    def __init__(self, sensorinfo, L1Bname, GEOpath=None):
        print("Reading Level 1B data ...")
        self.sensor = sensorinfo.sensor
        self.datasource = sensorinfo.datasource
        self.datalevel = sensorinfo.datalevel
        self.band = sensorinfo.band
        self.l1bvgain = sensorinfo.vgain
        self.l1bname = L1Bname
        if GEOpath == None:
            self.geopath = "./GEO/"
        else:
            self.geopath = GEOpath
        self.geoloc_status = 0
        self.lp_status = 0
        self.sline = 0
        self.eline = -1
        self.spixl = 0
        self.epixl = -1
        self.dim = np.zeros(2, dtype=int)
        if self.sensor == "OLI":
            # find MTL file
            l8mtl_path = glob(self.l1bname + "/*MTL.txt")[0]
            self.l8metadata = load_mtl(l8mtl_path)
        if self.sensor in ["S2A", "S2B"]:
            # read the xml file
            self.s2_path = (
                self.l1bname + "/GRANULE/" + listdir(self.l1bname + "/GRANULE/")[0]
            )
            xml_path = glob(self.s2_path + "/*.xml")[0]
            self.s2xmlroot = objectify.parse(xml_path).getroot()
            self.geocoding = self.s2xmlroot.Geometric_Info.find("Tile_Geocoding")
            self.tileangles = self.s2xmlroot.Geometric_Info.find("Tile_Angles")

    def readgeo(self):
        if self.sensor == "EPIC":
            f = h5py.File(self.l1bname, "r")
            data = np.array(f["Band" + str(self.band[0]) + "nm/Image"]).transpose()
            self.dim[0] = data.shape[0]
            self.dim[1] = data.shape[1]
            self.imagedim = data.shape
            # read geoloacation and geometry data
            self.latitude = np.array(
                f["Band443nm/Geolocation/Earth/Latitude"]
            ).transpose()
            self.longitude = np.array(
                f["Band443nm/Geolocation/Earth/Longitude"]
            ).transpose()
            f.close()
        elif self.sensor == "SGLI":
            f = h5py.File(self.l1bname, "r")
            data = f.get("/Image_data/Lt_VN01")
            self.dim[0] = data.shape[0]
            self.dim[1] = data.shape[1]
            self.imagedim = data.shape
            # read geolocation
            data = f.get("Geometry_data/Latitude")
            lat = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            latdim = lat.shape
            resample = data.attrs["Resampling_interval"]
            lat_grid_x = np.arange(0, latdim[0] * resample, resample)  # line
            lat_grid_y = np.arange(0, latdim[1] * resample, resample)  # pixl
            lat_grid_xt = np.arange(0, self.dim[0])
            lat_grid_yt = np.arange(0, self.dim[1])
            func = interpolate.interp2d(lat_grid_y, lat_grid_x, lat, kind="linear")
            self.latitude = func(lat_grid_yt, lat_grid_xt)

            data = f.get("Geometry_data/Longitude")
            lon = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            londim = lon.shape
            resample = data.attrs["Resampling_interval"]
            lon_grid_x = np.arange(0, londim[0] * resample, resample)  # line
            lon_grid_y = np.arange(0, londim[1] * resample, resample)  # pixl
            lon_grid_xt = np.arange(0, self.dim[0])
            lon_grid_yt = np.arange(0, self.dim[1])
            func = interpolate.interp2d(lon_grid_y, lon_grid_x, lon, kind="linear")
            self.longitude = func(lon_grid_yt, lon_grid_xt)
            f.close()
        elif self.sensor == "GOCI":
            self.goci_aux = "./auxdata/common/GOCI_auxdata.h5"
            f = h5py.File(self.goci_aux, "r")
            self.latitude = np.array(f["Latitude"])
            self.longitude = np.array(f["Longitude"])
            f.close()
            self.dim[0] = self.latitude.shape[0]
            self.dim[1] = self.latitude.shape[1]
            self.imagedim = self.latitude.shape
        elif self.sensor == "OLCI":
            # read latitude and longitude
            f = h5py.File(self.l1bname + "/geo_coordinates.nc", "r")
            data = f.get("latitude")
            scale = data.attrs["scale_factor"]
            self.latitude = np.array(data) * scale
            data = f.get("longitude")
            scale = data.attrs["scale_factor"]
            self.longitude = np.array(data) * scale
            self.dim[0] = self.latitude.shape[0]
            self.dim[1] = self.latitude.shape[1]
            self.imagedim = self.latitude.shape
            f.close()
        elif self.sensor == "VIIRS":
            if self.datasource == "NASA DAAC":
                f = SD(self.l1bname, SDC.READ)
                self.latitude = np.rot90(f.select("Latitude")[:, :], 2)
                self.longitude = np.rot90(f.select("Longitude")[:, :], 2)
                self.dim[0] = self.latitude.shape[0]
                self.dim[1] = self.latitude.shape[1]
                self.imagedim = self.latitude.shape
            elif self.datasource == "OBPG":
                self.geoname = (
                    self.geopath + basename(self.l1bname)[0:14] + ".GEO-M_SNPP.nc"
                )
                if exists(self.geoname):
                    f = h5py.File(self.geoname, "r")
                    data = f.get("geolocation_data/latitude")
                    self.latitude = np.rot90(np.array(data), 2)
                    data = f.get("geolocation_data/longitude")
                    self.longitude = np.rot90(np.array(data), 2)
                    self.dim[0] = self.latitude.shape[0]
                    self.dim[1] = self.latitude.shape[1]
                    self.imagedim = self.latitude.shape
                    f.close()
                else:
                    self.geoloc_status = 1
                    print(
                        "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                            self.geoname
                        )
                    )
                    print(
                        "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                            self.geopath
                        ),
                        "\033[m",
                    )
            elif self.datasource == "NOAA":
                print("function under development")
        elif self.sensor == "OLI":
            utm_zone = str(
                self.l8metadata["L1_METADATA_FILE"]["PROJECTION_PARAMETERS"]["UTM_ZONE"]
            )
            resol = int(
                self.l8metadata["L1_METADATA_FILE"]["PROJECTION_PARAMETERS"][
                    "GRID_CELL_SIZE_REFLECTIVE"
                ]
            )
            ul_x = self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                "CORNER_UL_PROJECTION_X_PRODUCT"
            ]
            ul_y = self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                "CORNER_UL_PROJECTION_Y_PRODUCT"
            ]
            lr_x = self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                "CORNER_LR_PROJECTION_X_PRODUCT"
            ]
            lr_y = self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                "CORNER_LR_PROJECTION_Y_PRODUCT"
            ]
            self.dim[0] = int(
                self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                    "REFLECTIVE_LINES"
                ]
            )
            self.dim[1] = int(
                self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                    "REFLECTIVE_SAMPLES"
                ]
            )
            if lr_x > ul_x:
                resolx = resol
            else:
                resolx = -resol
            if lr_y > ul_y:
                resoly = resol
            else:
                resoly = -resol
            proj = pyproj.Proj("+init=EPSG:326" + utm_zone)
            x, y = np.meshgrid(
                ul_x + resolx * np.arange(self.dim[1]),
                ul_y + resoly * np.arange(self.dim[0]),
            )
            self.longitude, self.latitude = proj(x, y, inverse=True)
            self.imagedim = self.latitude.shape

        elif self.sensor == "MODIS-Aqua":
            if self.datasource == "NASA DAAC":
                geo_matchstr = basename(self.l1bname)[9:26]
                s = glob(self.geopath + "MYD03." + geo_matchstr + "*.hdf")
                if len(s) > 0:
                    self.geoname = s[0]
                    f = SD(self.geoname, SDC.READ)
                    self.latitude = np.rot90(f.select("Latitude")[:, :], 2)
                    self.longitude = np.rot90(f.select("Longitude")[:, :], 2)
                    self.dim[0] = self.latitude.shape[0]
                    self.dim[1] = self.latitude.shape[1]
                    self.imagedim = self.latitude.shape
                else:
                    self.geoloc_status = 1
                    print(
                        "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                            self.geoname
                        )
                    )
                    print(
                        "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                            self.geopath
                        ),
                        "\033[m",
                    )

            elif self.datasource == "OBPG":
                self.geoname = self.geopath + basename(self.l1bname)[0:14] + ".GEO"
                if exists(self.geoname):
                    f = SD(self.geoname, SDC.READ)
                    self.latitude = np.rot90(f.select("Latitude")[:, :], 2)
                    self.longitude = np.rot90(f.select("Longitude")[:, :], 2)
                    self.dim[0] = self.latitude.shape[0]
                    self.dim[1] = self.latitude.shape[1]
                    self.imagedim = self.latitude.shape
                else:
                    self.geoloc_status = 1
                    print(
                        "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                            self.geoname
                        )
                    )
                    print(
                        "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                            self.geopath
                        ),
                        "\033[m",
                    )
        elif self.sensor == "MODIS-Terra":
            if self.datasource == "NASA DAAC":
                geo_matchstr = basename(self.l1bname)[9:26]
                s = glob(self.geopath + "MOD03." + geo_matchstr + "*.hdf")
                if len(s) > 0:
                    self.geoname = s[0]
                    f = SD(self.geoname, SDC.READ)
                    self.latitude = f.select("Latitude")[:, :]
                    self.longitude = f.select("Longitude")[:, :]
                    self.dim[0] = self.latitude.shape[0]
                    self.dim[1] = self.latitude.shape[1]
                    self.imagedim = self.latitude.shape
                else:
                    self.geoloc_status = 1
                    print(
                        "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                            self.geoname
                        )
                    )
                    print(
                        "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                            self.geopath
                        ),
                        "\033[m",
                    )

            elif self.datasource == "OBPG":
                self.geoname = self.geopath + basename(self.l1bname)[0:14] + ".GEO"
                if exists(self.geoname):
                    f = SD(self.geoname, SDC.READ)
                    self.latitude = np.rot90(f.select("Latitude")[:, :], 2)
                    self.longitude = np.rot90(f.select("Longitude")[:, :], 2)
                    self.dim[0] = self.latitude.shape[0]
                    self.dim[1] = self.latitude.shape[1]
                    self.imagedim = self.latitude.shape
                else:
                    self.geoloc_status = 1
                    print(
                        "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                            self.geoname
                        )
                    )
                    print(
                        "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                            self.geopath
                        ),
                        "\033[m",
                    )
        elif self.sensor in ["S2A", "S2B"]:
            self.s2_resolution = 60  # process S2 data at 60m resolution
            utm_code = self.geocoding.find("HORIZONTAL_CS_CODE").text
            for e in self.geocoding.findall("Geoposition"):
                if e.attrib["resolution"] == str(self.s2_resolution):
                    ul_x = int(e.find("ULX").text)
                    ul_y = int(e.find("ULY").text)
                    resolx = int(e.find("XDIM").text)
                    resoly = int(e.find("YDIM").text)
            for e in self.geocoding.findall("Size"):
                if e.attrib["resolution"] == str(self.s2_resolution):
                    self.dim[0] = int(e.find("NROWS").text)
                    self.dim[1] = int(e.find("NCOLS").text)
            proj = pyproj.Proj("+init={}".format(utm_code))
            x, y = np.meshgrid(
                ul_x + resolx * np.arange(self.dim[1]),
                ul_y + resoly * np.arange(self.dim[0]),
            )
            self.longitude, self.latitude = proj(x, y, inverse=True)
            self.imagedim = self.latitude.shape
        elif self.sensor == "MERSI2":
            self.geoname = self.geopath + basename(self.l1bname)[0:32] + "_GEO1K_MS.HDF"
            if exists(self.geoname):
                f = h5py.File(self.geoname, "r")
                data = f.get("Geolocation/Latitude")
                self.latitude = np.rot90(np.array(data), 2)
                data = f.get("Geolocation/Longitude")
                self.longitude = np.rot90(np.array(data), 2)
                self.dim[0] = self.latitude.shape[0]
                self.dim[1] = self.latitude.shape[1]
                self.imagedim = self.latitude.shape
                f.close()
            else:
                self.geoloc_status = 1
                print(
                    "\033[1;31;47mWARNING: Unable to locate the GEO file {}, processing terminated ... ".format(
                        self.geoname
                    )
                )
                print(
                    "\033[1;31;47mPleaes copy the missing GEO file to directory {} and rerun OC-SMART ... \n".format(
                        self.geopath
                    ),
                    "\033[m",
                )
        elif self.sensor == "HICO":
            f = h5py.File(self.l1bname, "r")
            self.latitude = np.array(f["/navigation/latitudes"]) + 0.008  # +0.008
            self.longitude = np.array(f["/navigation/longitudes"]) - 0.007  # -0.007
            self.dim[0] = self.latitude.shape[0]
            self.dim[1] = self.latitude.shape[1]
            self.imagedim = self.latitude.shape
            f.close()
        elif self.sensor == "SeaWiFS":
            f = h5py.File(self.l1bname, "r")
            self.latitude = np.array(f["/navigation_data/latitude"])
            self.longitude = np.array(f["/navigation_data/longitude"])
            self.dim[0] = self.latitude.shape[0]
            self.dim[1] = self.latitude.shape[1]
            self.imagedim = self.latitude.shape
            f.close()

    def readl1b(self):
        if self.sline < 0:
            self.sline = 0
            print("subimage North boundary out of image, reset to image boundary ...")
        if self.spixl < 0:
            self.spixl = 0
            print("subimage West boundary out of image, reset to image boundary ...")
        if self.eline > self.dim[0] or self.eline < 0:
            self.eline = self.dim[0]
            print("subimage South boundary out of image, reset to image boundary ...")
        if self.epixl > self.dim[1] or self.epixl < 0:
            self.epixl = self.dim[1]
            print("subimage East boundary out of image, reset to image boundary ...")
        self.dim[0] = self.eline - self.sline
        self.dim[1] = self.epixl - self.spixl
        #        print(self.sline,self.eline,self.spixl,self.epixl)

        # reset geolocation to subimage
        self.latitude = self.latitude[self.sline : self.eline, self.spixl : self.epixl]
        self.longitude = self.longitude[
            self.sline : self.eline, self.spixl : self.epixl
        ]

        nband = len(self.band)
        if self.sensor == "EPIC":
            refscale = np.array([2.685e-5, 8.34e-6, 6.66e-6, 9.30e-6, 1.435e-5])
            # read reflectance data
            f = h5py.File(self.l1bname, "r")
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            for i in range(nband):
                data = (
                    np.array(f["Band" + str(self.band[i]) + "nm/Image"]).transpose()
                    * refscale[i]
                    / np.pi
                )
                self.reflectance[:, :, i] = data[
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            self.reflectance[np.isinf(self.reflectance)] = -999.0

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            tmp1 = []  # tmp solz
            tmp2 = []  # tmp senz
            tmp3 = []  # tmp sola
            tmp4 = []  # tmp sena
            for i in range(nband):
                tmp1.append(
                    np.array(
                        f[
                            "Band"
                            + str(self.band[i])
                            + "nm/Geolocation/Earth/SunAngleZenith"
                        ]
                    ).transpose()
                )
                tmp2.append(
                    np.array(
                        f[
                            "Band"
                            + str(self.band[i])
                            + "nm/Geolocation/Earth/ViewAngleZenith"
                        ]
                    ).transpose()
                )
                tmp3.append(
                    np.array(
                        f[
                            "Band"
                            + str(self.band[i])
                            + "nm/Geolocation/Earth/SunAngleAzimuth"
                        ]
                    ).transpose()
                )
                tmp4.append(
                    np.array(
                        f[
                            "Band"
                            + str(self.band[i])
                            + "nm/Geolocation/Earth/ViewAngleAzimuth"
                        ]
                    ).transpose()
                )
            tmp1 = np.array(tmp1)
            tmp2 = np.array(tmp2)
            tmp3 = np.array(tmp3)
            tmp4 = np.array(tmp4)
            tmp1[np.isinf(tmp1)] = -999.0
            tmp2[np.isinf(tmp2)] = -999.0
            tmp3[np.isinf(tmp3)] = -999.0
            tmp4[np.isinf(tmp4)] = -999.0
            ind_inf = (np.mean(tmp3, axis=0) == -999.0) | (
                np.mean(tmp4, axis=0) == -999.0
            )
            relaz = tmp4 - 180 - tmp3
            relaz[relaz > 180.0] = 360 - relaz[relaz > 180.0]
            relaz[relaz < -180.0] = 360 + relaz[relaz < -180.0]
            solz = np.mean(tmp1, axis=0)
            senz = np.mean(tmp2, axis=0)
            relaz = np.mean(relaz, axis=0)
            relaz[ind_inf] = -999.0
            self.solz = solz[self.sline : self.eline, self.spixl : self.epixl]
            self.senz = senz[self.sline : self.eline, self.spixl : self.epixl]
            self.relaz = relaz[self.sline : self.eline, self.spixl : self.epixl]
            # L1b correction for geometry
            for i in range(nband):
                self.reflectance[:, :, i] = (
                    self.reflectance[:, :, i]
                    / np.cos(
                        np.deg2rad(
                            tmp1[i, self.sline : self.eline, self.spixl : self.epixl]
                        )
                    )
                    * np.cos(np.deg2rad(self.solz))
                )
            f.close()

        elif self.sensor == "SGLI":
            band_list = [
                "Lt_VN01",
                "Lt_VN02",
                "Lt_VN03",
                "Lt_VN04",
                "Lt_VN05",
                "Lt_VN06",
                "Lt_VN07",
                "Lt_VN10",
            ]
            f = h5py.File(self.l1bname, "r")
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            # read SGLI reflectance
            for i in range(nband):
                data_name = "/Image_data/" + band_list[i]
                data = f.get(data_name)
                ref = np.array(data)
                max_dn = data.attrs["Maximum_valid_DN"]
                slope = data.attrs["Slope"]
                offset = data.attrs["Offset"]
                #                cwl    = data.attrs['Center_wavelength']
                mask = data.attrs["Mask"]
                F0 = data.attrs["Band_weighted_TOA_solar_irradiance"]
                # assert int(np.around(cwl))==self.bands[i]
                mask_mtx = np.zeros([self.imagedim[0], self.imagedim[1]], dtype=int)
                mask_mtx[:, :] = mask
                ind_fill = ref > max_dn
                ref2 = np.bitwise_and(ref, mask_mtx) * slope + offset
                ref2[ind_fill] = -999.0
                self.reflectance[:, :, i] = (
                    ref2[self.sline : self.eline, self.spixl : self.epixl]
                    / F0
                    * self.l1bvgain[i]
                )
            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            # read geometry
            data = f.get("Geometry_data/Solar_zenith")
            solz = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            solzdim = solz.shape
            resample = data.attrs["Resampling_interval"]
            solz_grid_x = np.arange(0, solzdim[0] * resample, resample)  # line
            solz_grid_y = np.arange(0, solzdim[1] * resample, resample)  # pixl
            solz_grid_xt = np.arange(0, self.imagedim[0])
            solz_grid_yt = np.arange(0, self.imagedim[1])
            func = interpolate.interp2d(solz_grid_y, solz_grid_x, solz, kind="linear")
            isolz = func(solz_grid_yt, solz_grid_xt)
            self.solz = isolz[self.sline : self.eline, self.spixl : self.epixl]

            data = f.get("Geometry_data/Sensor_zenith")
            senz = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            senzdim = senz.shape
            resample = data.attrs["Resampling_interval"]
            senz_grid_x = np.arange(0, senzdim[0] * resample, resample)  # line
            senz_grid_y = np.arange(0, senzdim[1] * resample, resample)  # pixl
            senz_grid_xt = np.arange(0, self.imagedim[0])
            senz_grid_yt = np.arange(0, self.imagedim[1])
            func = interpolate.interp2d(senz_grid_y, senz_grid_x, senz, kind="linear")
            isenz = func(senz_grid_yt, senz_grid_xt)
            self.senz = isenz[self.sline : self.eline, self.spixl : self.epixl]

            data = f.get("Geometry_data/Solar_azimuth")
            sola = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            soladim = sola.shape
            resample = data.attrs["Resampling_interval"]
            sola_grid_x = np.arange(0, soladim[0] * resample, resample)  # line
            sola_grid_y = np.arange(0, soladim[1] * resample, resample)  # pixl
            sola_grid_xt = np.arange(0, self.imagedim[0])
            sola_grid_yt = np.arange(0, self.imagedim[1])
            func = interpolate.interp2d(sola_grid_y, sola_grid_x, sola, kind="linear")
            sola_interp = func(sola_grid_yt, sola_grid_xt)

            data = f.get("Geometry_data/Sensor_azimuth")
            sena = np.array(data) * data.attrs["Slope"] + data.attrs["Offset"]
            senadim = sena.shape
            resample = data.attrs["Resampling_interval"]
            sena_grid_x = np.arange(0, senadim[0] * resample, resample)  # line
            sena_grid_y = np.arange(0, senadim[1] * resample, resample)  # pixl
            sena_grid_xt = np.arange(0, self.imagedim[0])
            sena_grid_yt = np.arange(0, self.imagedim[1])
            func = interpolate.interp2d(sena_grid_y, sena_grid_x, sena, kind="linear")
            sena_interp = func(sena_grid_yt, sena_grid_xt)

            relaz = sena_interp - 180.0 - sola_interp
            relaz[relaz > 180.0] = 360.0 - relaz[relaz > 180.0]
            relaz[relaz < -180.0] = 360.0 + relaz[relaz < -180.0]
            self.relaz = relaz[self.sline : self.eline, self.spixl : self.epixl]

            data = f.get("/Image_data/Land_water_flag")
            self.landmask = np.array(
                data[self.sline : self.eline, self.spixl : self.epixl]
            )

            f.close()

        elif self.sensor == "GOCI":
            goci_f0 = [
                173.231,
                189.070,
                196.490,
                183.335,
                151.961,
                147.492,
                127.785,
                95.443,
            ]  # [mW/cm^2/um/sr]
            f = h5py.File(self.l1bname, "r")
            print("Generating GOCI slot data...")
            nav_data = f["HDFEOS/POINTS/Navigation for GOCI/Data/Navigation for GOCI"]
            self.goci_slot = goci_slots(
                nav_data, self.sline, self.eline, self.spixl, self.epixl, 7
            )
            goci_slot_relat_time = goci_slots_time(nav_data)
            print("Finished generating GOCI slot data.")
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            nslot = 16

            f_aux = h5py.File(self.goci_aux, "r")
            self.landmask = np.array(
                f_aux["Landmask"][self.sline : self.eline, self.spixl : self.epixl]
            )
            f_aux.close()

            # use centertime for esdist correction
            centertime = (
                f["HDFEOS/POINTS/Ephemeris"].attrs["Scene center time"].decode("utf-8")
            )
            dt = time.strptime(centertime, "%d-%b-%Y %H:%M:%S.%f")
            es_factor = es_dist(
                dt.tm_year, dt.tm_yday, dt.tm_hour * 3600 + dt.tm_min * 60 + dt.tm_sec
            )
            #            print(es_factor)
            scale = f["HDFEOS/POINTS/Radiometric Calibration for GOCI"].attrs[
                "Table for DN to Radiance conversion"
            ]  ##[W/m^2/um/sr]
            for i in np.arange(nband):
                data = (
                    np.array(
                        f[
                            "HDFEOS/GRIDS/Image Data/Data Fields/Band "
                            + str(i + 1)
                            + " Image Pixel Values"
                        ]
                    )
                    * scale[i]
                    / goci_f0[i]
                    / 10.0
                )
                if i == 0:
                    l1bmask = data
                else:
                    l1bmask = l1bmask * data
            l1bmask[l1bmask == 0] = -1
            l1bmask[l1bmask > 0] = 1
            f.close()
            l1bmask = l1bmask[self.sline : self.eline, self.spixl : self.epixl]
            self.goci_slot[l1bmask < 0] = -1
            basetimestr = basename(self.l1bname)[17:30]
            basedt = time.strptime(basetimestr, "%Y%m%d%H%M%S")
            basetime = calendar.timegm(basedt)
            solz = np.zeros(self.dim) - 999.0
            sola = np.zeros(self.dim) - 999.0
            senz = np.zeros(self.dim) - 999.0
            sena = np.zeros(self.dim) - 999.0
            for i in np.arange(nslot):
                realtime = basetime + round(goci_slot_relat_time[i])
                realdt = time.gmtime(realtime)
                #                print(realdt.tm_year,realdt.tm_mon,realdt.tm_mday,realdt.tm_hour,realdt.tm_min,realdt.tm_sec)
                (
                    solz[self.goci_slot == i + 1],
                    sola[self.goci_slot == i + 1],
                ) = goci_solar_angles(
                    realdt.tm_year,
                    realdt.tm_mon,
                    realdt.tm_mday,
                    realdt.tm_hour,
                    realdt.tm_min,
                    realdt.tm_sec,
                    self.latitude[self.goci_slot == i + 1],
                    self.longitude[self.goci_slot == i + 1],
                )
                senz[self.goci_slot == i + 1] = goci_sensor_zenith(
                    self.latitude[self.goci_slot == i + 1],
                    self.longitude[self.goci_slot == i + 1],
                )
                sena[self.goci_slot == i + 1] = goci_sensor_azimuth(
                    self.latitude[self.goci_slot == i + 1],
                    self.longitude[self.goci_slot == i + 1],
                    128.2,
                )
            self.solz = solz
            self.senz = senz
            l1bmask[(solz < 0) | (senz < 0)] = -1
            #            idx_inval=np.isnan(senz)==1
            relaz = sena - 180.0 - sola
            relaz[relaz > 180.0] = 360.0 - relaz[relaz > 180.0]
            relaz[relaz < -180.0] = 360.0 + relaz[relaz < -180.0]
            relaz[l1bmask < 0] = -999.0
            self.relaz = relaz

            f = h5py.File(self.l1bname, "r")
            for i in np.arange(nband):
                data = (
                    np.array(
                        f[
                            "HDFEOS/GRIDS/Image Data/Data Fields/Band "
                            + str(i + 1)
                            + " Image Pixel Values"
                        ][self.sline : self.eline, self.spixl : self.epixl]
                    )
                    * scale[i]
                    / goci_f0[i]
                    / 10.0
                    / es_factor
                    * self.l1bvgain[i]
                )
                data[l1bmask < 0] = -1
                # data = data[self.sline : self.eline, self.spixl : self.epixl]
                data[l1bmask < 0] = -1
                self.reflectance[:, :, i] = data / np.cos(np.deg2rad(self.solz))
            f.close()
            self.l1bmask = l1bmask

        elif self.sensor == "OLCI":
            band_list = [
                "Oa01",
                "Oa02",
                "Oa03",
                "Oa04",
                "Oa05",
                "Oa06",
                "Oa07",
                "Oa08",
                "Oa09",
                "Oa10",
                "Oa11",
                "Oa12",
                "Oa16",
                "Oa17",
                "Oa18",
                "Oa21",
            ]
            # read geometry and interpolate
            f = h5py.File(self.l1bname + "/tie_geometries.nc", "r")
            data = f.get("SZA")
            scale = data.attrs["scale_factor"]
            solz = np.array(data) * scale
            data = f.get("SAA")
            scale = data.attrs["scale_factor"]
            sola = np.array(data) * scale
            data = f.get("OZA")
            scale = data.attrs["scale_factor"]
            senz = np.array(data) * scale
            data = f.get("OAA")
            scale = data.attrs["scale_factor"]
            sena = np.array(data) * scale
            f.close()
            geodim = solz.shape
            dx = (self.imagedim[0] - 1) / (geodim[0] - 1)
            dy = (self.imagedim[1] - 1) / (geodim[1] - 1)
            grid_x = np.arange(0, self.imagedim[0], dx)
            grid_y = np.arange(0, self.imagedim[1], dy)
            grid_xt = np.arange(0, self.imagedim[0])
            grid_yt = np.arange(0, self.imagedim[1])
            func = interpolate.interp2d(grid_y, grid_x, solz, kind="linear")
            self.solz = func(grid_yt, grid_xt)
            func = interpolate.interp2d(grid_y, grid_x, sola, kind="linear")
            sola_interp = func(grid_yt, grid_xt)
            func = interpolate.interp2d(grid_y, grid_x, senz, kind="linear")
            self.senz = func(grid_yt, grid_xt)
            func = interpolate.interp2d(grid_y, grid_x, sena, kind="linear")
            sena_interp = func(grid_yt, grid_xt)

            self.relaz = sena_interp - 180.0 - sola_interp
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]

            self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
            self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
            self.relaz = self.relaz[self.sline : self.eline, self.spixl : self.epixl]

            # read radiance data
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            bid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20])
            f = h5py.File(self.l1bname + "/instrument_data.nc", "r")
            data = f.get("solar_flux")
            F0 = np.array(data)
            data = f.get("detector_index")
            di = np.array(data)
            f.close()

            for i in range(nband):
                band_F0 = np.zeros(self.imagedim)
                band_F0[:, :] = F0[bid[i], di]
                name = band_list[i] + "_radiance"
                f = h5py.File(self.l1bname + "/" + name + ".nc", "r")
                data = f.get(name)
                scale = data.attrs["scale_factor"]
                offset = data.attrs["add_offset"]
                ref = np.array(data)
                ref[ref > 65534] = 0
                f.close()
                ref = (ref * scale + offset) / band_F0 * self.l1bvgain[i]
                self.reflectance[:, :, i] = ref[
                    self.sline : self.eline, self.spixl : self.epixl
                ]

            # read landmask
            f = h5py.File(self.l1bname + "/qualityFlags.nc", "r")
            fmask = f.get("quality_flags").attrs["flag_masks"]
            fmeaning = str(
                f.get("quality_flags").attrs["flag_meanings"].decode("utf-8")
            ).split()
            quality_flags = {}
            for i in range(len(fmeaning)):
                quality_flags[fmeaning[i]] = fmask[i]

            flags = np.array(f.get("quality_flags"))
            self.landmask = np.zeros(self.imagedim, dtype="int8")
            self.landmask[flags & quality_flags["land"] != 0] = 1  # set land
            self.landmask[
                flags & quality_flags["fresh_inland_water"] != 0
            ] = 0  # recover inland water
            self.landmask = self.landmask[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

        elif self.sensor == "VIIRS":
            if self.datasource == "NASA DAAC":
                f = SD(self.l1bname, SDC.READ)
                self.solz = np.rot90(f.select("SolarZenithAngle")[:, :], 2)
                self.senz = np.rot90(f.select("SatelliteZenithAngle")[:, :], 2)
                sola = np.rot90(f.select("SolarAzimuthAngle")[:, :], 2)
                sena = np.rot90(f.select("SatelliteAzimuthAngle")[:, :], 2)
                self.relaz = sena - 180.0 - sola
                self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
                self.relaz[self.relaz < -180.0] = (
                    360.0 + self.relaz[self.relaz < -180.0]
                )

                self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
                self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
                self.relaz = self.relaz[
                    self.sline : self.eline, self.spixl : self.epixl
                ]

                # reset geolocation to subimage
                self.latitude = self.latitude[
                    self.sline : self.eline, self.spixl : self.epixl
                ]
                self.longitude = self.longitude[
                    self.sline : self.eline, self.spixl : self.epixl
                ]

                self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
                for i in range(nband):
                    data = f.select("Reflectance_M" + str(i + 1))
                    ref = np.rot90(data[:, :], 2)
                    attidx = data.attr("Scale").index()
                    scale = data.attr(attidx).get()
                    attidx = data.attr("Offset").index()
                    offset = data.attr(attidx).get()
                    ref[ref > 65527] = 0  # set fill values to 0
                    ref = ref * scale + offset
                    self.reflectance[:, :, i] = (
                        ref[self.sline : self.eline, self.spixl : self.epixl]
                        * self.l1bvgain[i]
                        / np.pi
                        * np.cos(np.deg2rad(self.solz))
                    )
            elif self.datasource == "OBPG":
                f = h5py.File(self.geoname, "r")
                data = f.get("geolocation_data/solar_zenith")
                scale = data.attrs["scale_factor"]
                offset = data.attrs["add_offset"]
                self.solz = np.rot90(np.array(data), 2) * scale + offset
                self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]

                data = f.get("geolocation_data/sensor_zenith")
                scale = data.attrs["scale_factor"]
                offset = data.attrs["add_offset"]
                self.senz = np.rot90(np.array(data), 2) * scale + offset
                self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]

                data = f.get("geolocation_data/solar_azimuth")
                scale = data.attrs["scale_factor"]
                offset = data.attrs["add_offset"]
                sola = np.rot90(np.array(data), 2) * scale + offset

                data = f.get("geolocation_data/sensor_azimuth")
                scale = data.attrs["scale_factor"]
                offset = data.attrs["add_offset"]
                sena = np.rot90(np.array(data), 2) * scale + offset

                self.relaz = sena - 180.0 - sola
                self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
                self.relaz[self.relaz < -180.0] = (
                    360.0 + self.relaz[self.relaz < -180.0]
                )
                self.relaz = self.relaz[
                    self.sline : self.eline, self.spixl : self.epixl
                ]

                self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
                f.close()

                # reset geolocation to subimage
                self.latitude = self.latitude[
                    self.sline : self.eline, self.spixl : self.epixl
                ]
                self.longitude = self.longitude[
                    self.sline : self.eline, self.spixl : self.epixl
                ]

                f = h5py.File(self.l1bname, "r")
                for i in range(nband):
                    data = f.get("observation_data/M" + "{:02d}".format(i + 1))
                    ref = np.rot90(np.array(data), 2)
                    scale = data.attrs["scale_factor"]
                    offset = data.attrs["add_offset"]
                    ref[ref > 65527] = 0  # set fill values to 0
                    self.reflectance[:, :, i] = (
                        (
                            ref[self.sline : self.eline, self.spixl : self.epixl]
                            * scale
                            + offset
                        )
                        * self.l1bvgain[i]
                        / np.pi
                    )
                f.close()

            elif self.datasource == "NOAA":
                print("function under development")
        elif self.sensor == "OLI":
            # read reflectanc of the first 5 bands 443 482 561 655 865
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            for i in np.arange(5):
                ds = gdal.Open(
                    self.l1bname
                    + "/"
                    + self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                        "FILE_NAME_BAND_" + str(i + 1)
                    ]
                )
                self.reflectance[:, :, i] = (
                    (
                        ds.ReadAsArray()[
                            self.sline : self.eline, self.spixl : self.epixl
                        ]
                        * self.l8metadata["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"][
                            "REFLECTANCE_MULT_BAND_" + str(i + 1)
                        ]
                        + self.l8metadata["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"][
                            "REFLECTANCE_ADD_BAND_" + str(i + 1)
                        ]
                    )
                    * self.l1bvgain[i]
                    / np.pi
                )
                ds = None
            ds = gdal.Open(
                self.l1bname
                + "/"
                + self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                    "FILE_NAME_BAND_QUALITY"
                ]
            )
            flags = ds.ReadAsArray()[self.sline : self.eline, self.spixl : self.epixl]
            ds = None
            cloud = np.bitwise_and(
                flags, int("101010000", 2)
            )  # maks cloud and could shadows
            self.cloud = cloud > 0
            angle_file = (
                self.l1bname
                + "/"
                + self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                    "ANGLE_COEFFICIENT_FILE_NAME"
                ]
            )

            angles = l8angles.calculate_angles(
                angle_file, angle_type="BOTH", bands=[1, 2, 3, 4, 5]
            )
            self.solz = np.zeros([self.dim[0], self.dim[1]])
            self.senz = np.zeros([self.dim[0], self.dim[1]])
            sola = np.zeros([self.dim[0], self.dim[1]])
            sena = np.zeros([self.dim[0], self.dim[1]])
            for i in np.arange(5):
                self.solz = (
                    self.solz
                    + angles["sun_zn"][i][
                        self.sline : self.eline, self.spixl : self.epixl
                    ]
                )
                self.senz = (
                    self.senz
                    + angles["sat_zn"][i][
                        self.sline : self.eline, self.spixl : self.epixl
                    ]
                )
                sola = (
                    sola
                    + angles["sun_az"][i][
                        self.sline : self.eline, self.spixl : self.epixl
                    ]
                )
                sena = (
                    sena
                    + angles["sat_az"][i][
                        self.sline : self.eline, self.spixl : self.epixl
                    ]
                )
            self.solz = self.solz / 5
            self.senz = self.senz / 5

            idx = np.isnan(self.solz)
            self.solz[idx] = -999.0
            idx = np.isnan(self.senz)
            self.senz[idx] = -999.0
            self.relaz = sena - 180.0 - sola
            idx = np.isnan(self.relaz)
            self.relaz[idx] = -999.0
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]
            self.relaz[idx] = np.nan

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            # apply gsw land/water mask
            self.landmask = GSW().get(self.latitude, self.longitude)

        elif self.sensor == "MODIS-Aqua":
            f = SD(self.geoname, SDC.READ)
            data = f.select("SolarZenith")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            self.solz = np.rot90(data[:, :], 2) * scale

            data = f.select("SensorZenith")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            self.senz = np.rot90(data[:, :], 2) * scale

            data = f.select("SolarAzimuth")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            sola = np.rot90(data[:, :], 2) * scale

            data = f.select("SensorAzimuth")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            sena = np.rot90(data[:, :], 2) * scale

            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]

            self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
            self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
            self.relaz = self.relaz[self.sline : self.eline, self.spixl : self.epixl]

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            f = SD(self.l1bname, SDC.READ)
            bandid = [0, 1, 2, 3, 4, 5, 7, 9, 10]
            data = f.select("EV_1KM_RefSB")
            attidx = data.attr("reflectance_scales").index()
            reflec_scale = data.attr(attidx).get()
            attidx = data.attr("reflectance_offsets").index()
            reflec_offset = data.attr(attidx).get()
            for i in np.arange(nband):
                tmp = data[bandid[i], :, :]
                tmp[tmp > 32767] = 0
                tmp = np.rot90(tmp, 2)
                self.reflectance[:, :, i] = (
                    (
                        tmp[self.sline : self.eline, self.spixl : self.epixl]
                        - reflec_offset[bandid[i]]
                    )
                    * reflec_scale[bandid[i]]
                    / np.pi
                    * self.l1bvgain[i]
                )

        elif self.sensor == "MODIS-Terra":
            f = SD(self.geoname, SDC.READ)
            data = f.select("SolarZenith")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            self.solz = data[:, :] * scale

            data = f.select("SensorZenith")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            self.senz = data[:, :] * scale

            data = f.select("SolarAzimuth")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            sola = data[:, :] * scale

            data = f.select("SensorAzimuth")
            attidx = data.attr("scale_factor").index()
            scale = data.attr(attidx).get()
            sena = data[:, :] * scale

            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]

            self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
            self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
            self.relaz = self.relaz[self.sline : self.eline, self.spixl : self.epixl]

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            f = SD(self.l1bname, SDC.READ)
            bandid = [0, 1, 2, 3, 4, 5, 7, 9, 10]
            data = f.select("EV_1KM_RefSB")
            attidx = data.attr("reflectance_scales").index()
            reflec_scale = data.attr(attidx).get()
            attidx = data.attr("reflectance_offsets").index()
            reflec_offset = data.attr(attidx).get()
            for i in np.arange(nband):
                tmp = data[bandid[i], :, :]
                tmp[tmp > 32767] = 0
                self.reflectance[:, :, i] = (
                    (
                        tmp[self.sline : self.eline, self.spixl : self.epixl]
                        - reflec_offset[bandid[i]]
                    )
                    * reflec_scale[bandid[i]]
                    / np.pi
                    * self.l1bvgain[i]
                )

        elif self.sensor in ["S2A", "S2B"]:
            band_list = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A"]

            # read solar angle
            sza = read_xml_block(
                self.tileangles.find("Sun_Angles_Grid")
                .find("Zenith")
                .find("Values_List")
            )
            saa = read_xml_block(
                self.tileangles.find("Sun_Angles_Grid")
                .find("Azimuth")
                .find("Values_List")
            )
            self.solz = rectBivariateSpline(sza, self.imagedim)
            self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
            sola = rectBivariateSpline(saa, self.imagedim)

            # read sensor angles (for each band)
            vza = {}
            vaa = {}
            for e in self.tileangles.findall("Viewing_Incidence_Angles_Grids"):
                # read zenith angles
                data = read_xml_block(e.find("Zenith").find("Values_List"))
                bandid = int(e.attrib["bandId"])
                if not bandid in vza:
                    vza[bandid] = data
                else:
                    ok = ~np.isnan(data)
                    vza[bandid][ok] = data[ok]
                # read azimuth angles
                data = read_xml_block(e.find("Azimuth").find("Values_List"))
                bandid = int(e.attrib["bandId"])
                if not bandid in vaa:
                    vaa[bandid] = data
                else:
                    ok = ~np.isnan(data)
                    vaa[bandid][ok] = data[ok]
            # use the mean value of band 0-8 as senz and sena
            vza_mean = np.mean(np.asarray(list(vza.values()))[0:nband, :, :], axis=0)
            vaa_mean = np.mean(np.asarray(list(vaa.values()))[0:nband, :, :], axis=0)
            self.senz = rectBivariateSpline(vza_mean, self.imagedim)
            self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
            sena = rectBivariateSpline(vaa_mean, self.imagedim)
            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]
            self.relaz = self.relaz[self.sline : self.eline, self.spixl : self.epixl]

            # read image data and convert to reflectance
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            QUANTIFICATION = 10000
            if self.datalevel == "L1C":
                for i in np.arange(nband):
                    fid = glob(self.s2_path + "/IMG_DATA/*{}.jp2".format(band_list[i]))[
                        0
                    ]
                    jp = Jp2k(fid)
                    ratiox = jp.shape[0] / self.imagedim[0]
                    ratioy = jp.shape[1] / self.imagedim[1]
                    #                print(ratiox,ratioy)
                    if ratiox > 1 or ratioy > 1:
                        data = downscale(jp[:, :], self.imagedim)
                    else:
                        data = jp[:, :]
                    self.reflectance[:, :, i] = (
                        data[self.sline : self.eline, self.spixl : self.epixl]
                        / QUANTIFICATION
                        * self.l1bvgain[i]
                        / np.pi
                        * np.cos(np.deg2rad(self.solz))
                    )
            elif self.datalevel == "L2A":
                for i in np.arange(nband - 2):
                    fid = glob(
                        self.s2_path + "/IMG_DATA/R60m/*{}*.jp2".format(band_list[i])
                    )[0]
                    jp = Jp2k(fid)
                    ratiox = jp.shape[0] / self.imagedim[0]
                    ratioy = jp.shape[1] / self.imagedim[1]
                    #                print(ratiox,ratioy)
                    if ratiox > 1 or ratioy > 1:
                        data = downscale(jp[:, :], self.imagedim)
                    else:
                        data = jp[:, :]
                    self.reflectance[:, :, i] = (
                        data[self.sline : self.eline, self.spixl : self.epixl]
                        / QUANTIFICATION
                        * self.l1bvgain[i]
                        / np.pi
                        * np.cos(np.deg2rad(self.solz))
                    )

                fid = glob(
                    self.s2_path + "/IMG_DATA/R10m/*{}*.jp2".format(band_list[7])
                )[0]
                jp = Jp2k(fid)
                ratiox = jp.shape[0] / self.imagedim[0]
                ratioy = jp.shape[1] / self.imagedim[1]
                #                print(ratiox,ratioy)
                if ratiox > 1 or ratioy > 1:
                    data = downscale(jp[:, :], self.imagedim)
                else:
                    data = jp[:, :]
                self.reflectance[:, :, 7] = (
                    data[self.sline : self.eline, self.spixl : self.epixl]
                    / QUANTIFICATION
                    * self.l1bvgain[7]
                    / np.pi
                    * np.cos(np.deg2rad(self.solz))
                )
                fid = glob(
                    self.s2_path + "/IMG_DATA/R60m/*{}*.jp2".format(band_list[8])
                )[0]
                jp = Jp2k(fid)
                ratiox = jp.shape[0] / self.imagedim[0]
                ratioy = jp.shape[1] / self.imagedim[1]
                #                print(ratiox,ratioy)
                if ratiox > 1 or ratioy > 1:
                    data = downscale(jp[:, :], self.imagedim)
                else:
                    data = jp[:, :]
                self.reflectance[:, :, 8] = (
                    data[self.sline : self.eline, self.spixl : self.epixl]
                    / QUANTIFICATION
                    * self.l1bvgain[8]
                    / np.pi
                    * np.cos(np.deg2rad(self.solz))
                )

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            # apply gsw land/water mask
            self.landmask = GSW().get(self.latitude, self.longitude)
        elif self.sensor == "MERSI2":
            f = h5py.File(self.geoname, "r")
            data = f.get("Geolocation/SolarZenith")
            scale = data.attrs["Slope"]
            self.solz = np.rot90(np.array(data), 2) * scale

            data = f.get("Geolocation/SensorZenith")
            scale = data.attrs["Slope"]
            self.senz = np.rot90(np.array(data), 2) * scale

            data = f.get("Geolocation/SolarAzimuth")
            scale = data.attrs["Slope"]
            sola = np.rot90(np.array(data), 2) * scale

            data = f.get("Geolocation/SensorAzimuth")
            scale = data.attrs["Slope"]
            sena = np.rot90(np.array(data), 2) * scale
            f.close()

            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]

            self.solz = self.solz[self.sline : self.eline, self.spixl : self.epixl]
            self.senz = self.senz[self.sline : self.eline, self.spixl : self.epixl]
            self.relaz = self.relaz[self.sline : self.eline, self.spixl : self.epixl]

            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            bandid = np.arange(3, 11)  # band 8 - 15
            f = h5py.File(self.l1bname, "r")
            scale = 0.01
            data = f.get("Calibration/VIS_Cal_Coeff")
            mersi_vis_cal = np.array(data)
            data = np.array(f.get("Data/EV_1KM_RefSB"))
            for i in np.arange(nband):
                tmp = data[bandid[i], :, :]
                tmp[tmp > 4095] = 0
                tmp = np.rot90(tmp, 2)
                self.reflectance[:, :, i] = scale * (
                    mersi_vis_cal[bandid[i] + 4, 0]
                    + mersi_vis_cal[bandid[i] + 4, 1]
                    * tmp[self.sline : self.eline, self.spixl : self.epixl]
                    + mersi_vis_cal[bandid[i] + 4, 2]
                    * pow(tmp[self.sline : self.eline, self.spixl : self.epixl], 2)
                )
                self.reflectance[:, :, i] = (
                    self.reflectance[:, :, i] / np.pi * self.l1bvgain[i]
                )
            f.close()
        elif self.sensor == "HICO":
            bandlist = np.zeros(
                78
            )  # 404-896nm but exclude 719,725,730,759,765,770,816,821,827nm(watervapor and o2 absorption bands)
            bandlist[0:55] = np.asarray(np.arange(9, 64))  # 404-713nm
            bandlist[55:59] = np.asarray(np.arange(67, 71))  # 736-752nm
            bandlist[59:66] = np.asarray(np.arange(74, 81))  # 776-810nm
            bandlist[66:78] = np.asarray(np.arange(84, 96))  # 833-896nm
            F0 = np.loadtxt("./auxdata/sensorinfo/HICO_F0.txt")  # mW/cm^2/um/sr
            f = h5py.File(self.l1bname, "r")
            self.solz = np.array(
                f["navigation/solar_zenith"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            self.senz = np.array(
                f["navigation/sensor_zenith"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            sola = np.array(
                f["navigation/solar_azimuth"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            sena = np.array(
                f["navigation/sensor_azimuth"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            rad = np.array(
                f["products/Lt"][
                    self.sline : self.eline, self.spixl : self.epixl, bandlist
                ]
            )
            rad_tot = np.sum(rad, axis=2)
            scale = np.array(f["products/Lt"].attrs["scale_factor"])
            offset = np.array(f["products/Lt"].attrs["add_offset"])
            for i in np.arange(nband):
                tmp = rad[:, :, i]
                tmp[rad_tot == 0] = -1  # image border
                tmp[tmp == 0] = np.max(tmp)  # possible satuation
                self.reflectance[:, :, i] = (
                    (tmp * scale + offset) / F0[i] / 10.0 * self.l1bvgain[i]
                )
            f.close()
            # apply gsw land/water mask
            self.landmask = GSW().get(self.latitude, self.longitude)
            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
        elif self.sensor == "SeaWiFS":
            f0 = [
                172.998,
                190.154,
                196.438,
                188.164,
                182.997,
                151.139,
                122.330,
                96.264,
            ]  # [mW/cm^2/um/sr]
            band_list = [
                "Lt_412",
                "Lt_443",
                "Lt_490",
                "Lt_510",
                "Lt_555",
                "Lt_670",
                "Lt_765",
                "Lt_865",
            ]
            dt = time.strptime(basename(self.l1bname)[1:14], "%Y%j%H%M%S")
            es_factor = es_dist(
                dt.tm_year, dt.tm_yday, dt.tm_hour * 3600 + dt.tm_min * 60 + dt.tm_sec
            )
            f = h5py.File(self.l1bname, "r")
            self.solz = np.array(
                f["navigation_data/solz"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            scale = np.array(f["navigation_data/solz"].attrs["scale_factor"])
            offset = np.array(f["navigation_data/solz"].attrs["add_offset"])
            self.solz = self.solz * scale + offset
            self.senz = np.array(
                f["navigation_data/senz"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            scale = np.array(f["navigation_data/senz"].attrs["scale_factor"])
            offset = np.array(f["navigation_data/senz"].attrs["add_offset"])
            self.senz = self.senz * scale + offset
            sola = np.array(
                f["navigation_data/sola"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            scale = np.array(f["navigation_data/sola"].attrs["scale_factor"])
            offset = np.array(f["navigation_data/sola"].attrs["add_offset"])
            sola = sola * scale + offset
            sena = np.array(
                f["navigation_data/sena"][
                    self.sline : self.eline, self.spixl : self.epixl
                ]
            )
            scale = np.array(f["navigation_data/sena"].attrs["scale_factor"])
            offset = np.array(f["navigation_data/sena"].attrs["add_offset"])
            sena = sena * scale + offset
            self.relaz = sena - 180.0 - sola
            self.relaz[self.relaz > 180.0] = 360.0 - self.relaz[self.relaz > 180.0]
            self.relaz[self.relaz < -180.0] = 360.0 + self.relaz[self.relaz < -180.0]
            self.reflectance = np.zeros([self.dim[0], self.dim[1], nband])
            for i in np.arange(nband):
                self.reflectance[:, :, i] = (
                    np.array(
                        f["geophysical_data/" + band_list[i]][
                            self.sline : self.eline, self.spixl : self.epixl
                        ]
                    )
                    / f0[i]
                    / es_factor
                    / 10.0
                    * self.l1bvgain[i]
                )
            # reset geolocation to subimage
            self.latitude = self.latitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]
            self.longitude = self.longitude[
                self.sline : self.eline, self.spixl : self.epixl
            ]

    def latlon2linepixl(self, **kwargs):
        if "north" in kwargs:
            north = kwargs["north"]
        if "south" in kwargs:
            south = kwargs["south"]
        if "east" in kwargs:
            east = kwargs["east"]
        if "west" in kwargs:
            west = kwargs["west"]
        if "lat_center" in kwargs:
            lat_center = kwargs["lat_center"]
        if "lon_center" in kwargs:
            lon_center = kwargs["lon_center"]
        if "box_width" in kwargs:
            box_width = kwargs["box_width"]
        if "box_height" in kwargs:
            box_height = kwargs["box_height"]
        if "start_line" in kwargs:
            sline = kwargs["start_line"]
        if "end_line" in kwargs:
            eline = kwargs["end_line"]
        if "start_pixel" in kwargs:
            spixl = kwargs["start_pixel"]
        if "end_pixel" in kwargs:
            epixl = kwargs["end_pixel"]

        # mode 1: find line and pixel number by (north, south, east, west) coordinates
        if (
            "north" in locals()
            and "south" in locals()
            and "east" in locals()
            and "west" in locals()
        ):
            # make sure the input make sense
            if north > 90.0:
                print("Error: North coordinate:{} is invalid".format(north))
                self.lp_status = 1
            elif south < -90.0:
                print("Error: South coordinate:{} is invalid".format(south))
                self.lp_status = 1
            elif east > 180.0:
                print("Error: East coordinate:{} is invalid".format(east))
                self.lp_status = 1
            elif east < -180.0:
                print("Error: West coordinate:{} is invalid".format(west))
                self.lp_status = 1
            elif south > north:
                print(
                    "Error: South coordinate:{} is larger than North coordinate:{}".format(
                        south, north
                    )
                )
                self.lp_status = 1
            elif west > 0.0 and east > 0.0 and west > east:
                print(
                    "Error: West coordinate:{} is larger than East coordinate:{}".format(
                        west, east
                    )
                )
                self.lp_status = 1
            elif west < 0.0 and east < 0.0 and west > east:
                print(
                    "Error: West coordinate:{} is larger than East coordinate:{}".format(
                        west, east
                    )
                )
                self.lp_status = 1
            else:
                dist = (
                    (self.latitude - north) ** 2 + (self.longitude - west) ** 2
                ) ** 0.5
                self.sline = np.where(dist == np.min(dist))[0][0]
                self.spixl = np.where(dist == np.min(dist))[1][0]
                dist = (
                    (self.latitude - south) ** 2 + (self.longitude - east) ** 2
                ) ** 0.5
                self.eline = np.where(dist == np.min(dist))[0][0]
                self.epixl = np.where(dist == np.min(dist))[1][0]
                if self.sline == self.eline or self.spixl == self.epixl:
                    print(
                        "Coordinates out of image boundary, unable to extract a subimage..."
                    )
                    self.lp_status = 1

        # mode 2: find line and pixel number by (lat_center, lon_center, box_width, box_height) coordinates
        elif (
            "lat_center" in locals()
            and "lon_center" in locals()
            and "box_width" in locals()
            and "box_height" in locals()
        ):
            # make sure the input make sense
            if lat_center > 90.0 or lat_center < -90.0:
                print(
                    "Error: center latitute coordinate:{} is invalid".format(lat_center)
                )
                self.lp_status = 1
            elif lon_center > 180.0 or lon_center < -180.0:
                print(
                    "Error: center longitude coordinate:{} is invalid".format(
                        lon_center
                    )
                )
                self.lp_status = 1
            elif box_width < 0:
                print("Error: box width:{} is invalid".format(box_width))
                self.lp_status = 1
            elif box_height < 0:
                print("Error: box height:{} is invalid".format(box_height))
                self.lp_status = 1
            else:
                dist = (
                    (self.latitude - lat_center) ** 2
                    + (self.longitude - lon_center) ** 2
                ) ** 0.5
                nearest_line = np.where(dist == np.min(dist))[0][0]
                nearest_pixl = np.where(dist == np.min(dist))[1][0]
                if (
                    nearest_line == 0
                    or nearest_pixl == 0
                    or nearest_line + 1 == self.dim[0]
                    or nearest_pixl + 1 == self.dim[1]
                ):
                    print(
                        "Center lat/lon out of image boundary, unable to extract a subimage..."
                    )
                    self.lp_status = 1
                else:
                    self.sline = np.where(dist == np.min(dist))[0][0] - int(
                        box_height / 2
                    )
                    self.spixl = np.where(dist == np.min(dist))[1][0] - int(
                        box_width / 2
                    )
                    self.eline = (
                        np.where(dist == np.min(dist))[0][0] + int(box_height / 2) + 1
                    )
                    self.epixl = (
                        np.where(dist == np.min(dist))[1][0] + int(box_width / 2) + 1
                    )

        elif (
            "sline" in locals()
            and "eline" in locals()
            and "spixl" in locals()
            and "epixl" in locals()
        ):
            self.sline = sline
            self.eline = eline
            self.spixl = spixl
            self.epixl = epixl
        else:
            print(
                "Missing Parameters, subimage should be defined by: (north, south, east, west) or (lat_center, lon_center, box_width, box_height) or (start_line, end_line, start_pixel, end_pixel)"
            )
            self.lp_status = 1


# util functions
def read_xml_block(item):
    # read a block of xml data and returns it as a numpy float32 array
    d = []
    for i in item.iterchildren():
        d.append(i.text.split())
    return np.array(d, dtype="float32")


def downscale(a, shape):
    # downscale the high resolution data
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def rectBivariateSpline(A, shp):
    # Bivariate spline interpolation of array A to shape shp.
    # Fill NaNs with closest values, otherwise RectBivariateSpline gives no result.
    xin = np.arange(shp[0], dtype="float32") / (shp[0] - 1) * A.shape[0]
    yin = np.arange(shp[1], dtype="float32") / (shp[1] - 1) * A.shape[1]

    x = np.arange(A.shape[0], dtype="float32")
    y = np.arange(A.shape[1], dtype="float32")

    invalid = np.isnan(A)
    if invalid.any():
        # fill nans
        # see http://stackoverflow.com/questions/3662361/
        ind = distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        A = A[tuple(ind)]

    f = RectBivariateSpline(x, y, A)

    return f(xin, yin).astype("float32")


def es_dist(year, day, sec):
    jd = 367 * year - 7 * (year + 10 / 12) / 4 + 275 * 1 / 9 + day + 1721014
    t = jd - 2451545.0 + (sec - 43200) / 86400
    gs = 357.52772 + 0.9856002831 * t
    esdist = (
        1.00014
        - 0.01671 * np.cos(np.deg2rad(gs))
        - 0.00014 * np.cos(np.deg2rad(2 * gs))
    )
    return np.power((1 / esdist), 2)
