#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:05:04 2019

@author: Yongzhen Fan
"""
import bz2
import calendar
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from glob import glob
from os import listdir, makedirs
from os.path import basename, dirname, exists, isdir

import h5py
import numpy as np
from lxml import objectify
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from scipy import interpolate

from src.l8_utils import load_mtl
from src.obdaac_download import httpdl


class ANCILLARY(object):
    def __init__(self, sensorinfo, L1Bname):
        # find time info from L1Bname
        self.sensor = sensorinfo.sensor
        self.datasource = sensorinfo.datasource
        self.l1bpath = dirname(L1Bname)
        self.l1bname = basename(L1Bname)
        self.path = "./anc/"
        print("Locating ancillary files ...")
        if not isdir(self.path):
            print(
                "Ancillary directory does not exist, creating ancillary directory {} ... ".format(
                    self.path
                )
            )
            makedirs(self.path)
        if self.sensor == "EPIC":
            self.datestr = self.l1bname[8:22]
            self.year = self.l1bname[8:12]
            self.month = self.l1bname[12:14]
            self.day = self.l1bname[14:16]
            # self.l1btime=datetime.strptime(self.datestr,'%Y%m%d%H%M%S').timestamp()
        elif self.sensor == "SGLI":
            self.year = self.l1bname[7:11]
            self.month = self.l1bname[11:13]
            self.day = self.l1bname[13:15]
            f = h5py.File(self.l1bpath + "/" + self.l1bname, "r")
            obstime = np.array(f["Geometry_data/Obs_time"]) * 0.001
            midobstime = (obstime.min() + obstime.max()) / 2
            hour = "{:02d}".format(int(np.floor(midobstime)))
            mintime = round((midobstime - np.floor(midobstime)) * 60, 6)
            minute = "{:02d}".format(int(np.floor(mintime)))
            second = "{:02d}".format(int((mintime - np.floor(mintime)) * 60))
            self.datestr = self.l1bname[7:15] + hour + minute + second
            f.close()
            # self.l1btime=datetime.strptime(self.datestr,'%Y%m%d%H%M%S').timestamp()
        elif self.sensor == "VIIRS":
            if self.datasource == "NASA DAAC":
                self.year = self.l1bname[14:18]
                doy = int(self.l1bname[18:21])
                dt = datetime(int(self.year), 1, 1) + timedelta(doy - 1)
                self.month = "{:02d}".format(dt.month)
                self.day = "{:02d}".format(dt.day)
                self.datestr = (
                    self.year + self.month + self.day + self.l1bname[22:26] + "00"
                )
            elif self.datasource == "OBPG":
                self.year = self.l1bname[1:5]
                doy = int(self.l1bname[5:8])
                dt = datetime(int(self.year), 1, 1) + timedelta(doy - 1)
                self.month = "{:02d}".format(dt.month)
                self.day = "{:02d}".format(dt.day)
                self.datestr = self.year + self.month + self.day + self.l1bname[8:14]
        elif self.sensor == "OLCI":
            posn = self.l1bname.find("____") + 4
            self.year = self.l1bname[posn : posn + 4]
            self.month = self.l1bname[posn + 4 : posn + 6]
            self.day = self.l1bname[posn + 6 : posn + 8]
            f = h5py.File(
                self.l1bpath + "/" + self.l1bname + "/time_coordinates.nc", "r"
            )
            obstime = (
                np.array(f["time_stamp"]) * 1.0e-6
            )  # Elapsed time since 01 Jan 2000 0h in seconds
            midobstime = np.round((obstime.min() + obstime.max()) / 2)
            dt = time.gmtime(
                calendar.timegm(time.strptime("200001010000", "%Y%m%d%H%M%S"))
                + midobstime
            )
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second
            f.close()
        elif self.sensor in ["MODIS-Aqua", "MODIS-Terra"]:
            if self.datasource == "NASA DAAC":
                self.year = self.l1bname[10:14]
                doy = int(self.l1bname[14:17])
                dt = datetime(int(self.year), 1, 1) + timedelta(doy - 1)
                self.month = "{:02d}".format(dt.month)
                self.day = "{:02d}".format(dt.day)
                self.datestr = (
                    self.year + self.month + self.day + self.l1bname[18:22] + "00"
                )
            elif self.datasource == "OBPG":
                self.year = self.l1bname[1:5]
                doy = int(self.l1bname[5:8])
                dt = datetime(int(self.year), 1, 1) + timedelta(doy - 1)
                self.month = "{:02d}".format(dt.month)
                self.day = "{:02d}".format(dt.day)
                self.datestr = self.year + self.month + self.day + self.l1bname[8:14]
        elif self.sensor == "OLI":
            # find MTL file
            l8mtl_path = glob(self.l1bpath + "/" + self.l1bname + "/*MTL.txt")[0]
            self.l8metadata = load_mtl(l8mtl_path)
            idx = self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                "SCENE_CENTER_TIME"
            ].find(".")
            dt = time.strptime(
                self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"]["DATE_ACQUIRED"]
                + self.l8metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"][
                    "SCENE_CENTER_TIME"
                ][0:idx],
                "%Y-%m-%d%H:%M:%S",
            )
            self.year = "{:04d}".format(dt.tm_year)
            self.month = "{:02d}".format(dt.tm_mon)
            self.day = "{:02d}".format(dt.tm_mday)
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second
        elif self.sensor in ["S2A", "S2B"]:
            # read the xml file
            self.s2_path = (
                self.l1bpath
                + "/"
                + self.l1bname
                + "/GRANULE/"
                + listdir(self.l1bpath + "/" + self.l1bname + "/GRANULE/")[0]
            )
            xml_path = glob(self.s2_path + "/*.xml")[0]
            self.s2xmlroot = objectify.parse(xml_path).getroot()
            dt = time.strptime(
                str(self.s2xmlroot.General_Info.find("SENSING_TIME")),
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
            self.year = "{:04d}".format(dt.tm_year)
            self.month = "{:02d}".format(dt.tm_mon)
            self.day = "{:02d}".format(dt.tm_mday)
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second
        elif self.sensor == "MERSI2":
            f = h5py.File(self.l1bpath + "/" + self.l1bname, "r")
            date_start = f.attrs["Observing Beginning Date"].decode("utf-8")
            date_end = f.attrs["Observing Ending Date"].decode("utf-8")
            time_start = f.attrs["Observing Beginning Time"].decode("utf-8")
            time_end = f.attrs["Observing Ending Time"].decode("utf-8")
            f.close()
            dt_start = time.strptime(date_start + time_start, "%Y-%m-%d%H:%M:%S.%f")
            dt_end = time.strptime(date_end + time_end, "%Y-%m-%d%H:%M:%S.%f")
            dt = time.gmtime((calendar.timegm(dt_start) + calendar.timegm(dt_end)) / 2)
            self.year = "{:04d}".format(dt.tm_year)
            self.month = "{:02d}".format(dt.tm_mon)
            self.day = "{:02d}".format(dt.tm_mday)
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second
        elif self.sensor == "GOCI":
            f = h5py.File(self.l1bpath + "/" + self.l1bname, "r")
            # use centertime for ancillary files, difference between centertime and slot 1&16 is less than 15 min
            centertime = (
                f["HDFEOS/POINTS/Ephemeris"].attrs["Scene center time"].decode("utf-8")
            )
            dt = time.strptime(centertime, "%d-%b-%Y %H:%M:%S.%f")
            f.close()
            self.year = "{:04d}".format(dt.tm_year)
            self.month = "{:02d}".format(dt.tm_mon)
            self.day = "{:02d}".format(dt.tm_mday)
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second
        elif self.sensor == "HICO":
            if self.datasource == "OBPG":
                self.year = self.l1bname[1:5]
                doy = int(self.l1bname[5:8])
                dt = datetime(int(self.year), 1, 1) + timedelta(doy - 1)
                self.month = "{:02d}".format(dt.month)
                self.day = "{:02d}".format(dt.day)
                self.datestr = self.year + self.month + self.day + self.l1bname[8:14]
        elif self.sensor == "SeaWiFS":
            dt = time.strptime(self.l1bname[1:14], "%Y%j%H%M%S")
            self.year = "{:04d}".format(dt.tm_year)
            self.month = "{:02d}".format(dt.tm_mon)
            self.day = "{:02d}".format(dt.tm_mday)
            hour = "{:02d}".format(dt.tm_hour)
            minute = "{:02d}".format(dt.tm_min)
            second = "{:02d}".format(dt.tm_sec)
            self.datestr = self.year + self.month + self.day + hour + minute + second

    def read_no2(self):
        auxpath = "./auxdata/common/"
        print("Reading NO2 data ...")
        # read NO2 data
        months = range(1, 13)
        nmonths = 12
        # set latitude and longitude grid
        self.no2_frac_lat = np.arange(91, -93, -2)
        self.no2_frac_lon = np.arange(-181, 183, 2)
        self.no2_lat = np.arange(90.125, -90.375, -0.25)
        self.no2_lon = np.arange(-180.125, 180.375, 0.25)
        no2_frac_nline = len(self.no2_frac_lat)
        no2_frac_npixl = len(self.no2_frac_lon)
        no2_nline = len(self.no2_lat)
        no2_npixl = len(self.no2_lon)

        self.no2_total = np.zeros((nmonths, no2_nline, no2_npixl), dtype="float64")
        self.no2_tropo = np.zeros((nmonths, no2_nline, no2_npixl), dtype="float64")
        self.no2_strat = np.zeros((nmonths, no2_nline, no2_npixl), dtype="float64")
        self.no2_frac = np.zeros((no2_frac_nline, no2_frac_npixl), dtype="float64")

        no2_fname = auxpath + "no2_climatology_v2013.hdf"
        no2_frac_fname = auxpath + "trop_f_no2_200m.hdf"

        # read no2 fraction data
        f = SD(no2_frac_fname, SDC.READ)
        self.no2_frac[1 : no2_frac_nline - 1, 1 : no2_frac_npixl - 1] = f.select(
            "f_no2_200m"
        )[:, :]
        self.no2_frac[:, 0] = self.no2_frac[:, no2_frac_npixl - 2]
        self.no2_frac[:, no2_frac_npixl - 1] = self.no2_frac[:, 1]
        self.no2_frac[0, :] = self.no2_frac[1, :]
        self.no2_frac[no2_frac_nline - 1, :] = self.no2_frac[no2_frac_nline - 2, :]

        # read total and tropospheric no2 data
        f = SD(no2_fname, SDC.READ)
        for i, m in enumerate(months):
            self.no2_tropo[i, 1 : no2_nline - 1, 1 : no2_npixl - 1] = f.select(
                "trop_no2_{:02d}".format(m)
            )[:, :]
            self.no2_tropo[i, :, 0] = self.no2_tropo[i, :, no2_npixl - 2]
            self.no2_tropo[i, :, no2_npixl - 1] = self.no2_tropo[i, :, 1]
            self.no2_tropo[i, 0, :] = self.no2_tropo[i, 1, :]
            self.no2_tropo[i, no2_nline - 1, :] = self.no2_tropo[i, no2_nline - 2, :]
            self.no2_total[i, 1 : no2_nline - 1, 1 : no2_npixl - 1] = f.select(
                "tot_no2_{:02d}".format(m)
            )[:, :]
            self.no2_total[i, :, 0] = self.no2_total[i, :, no2_npixl - 2]
            self.no2_total[i, :, no2_npixl - 1] = self.no2_total[i, :, 1]
            self.no2_total[i, 0, :] = self.no2_total[i, 1, :]
            self.no2_total[i, no2_nline - 1, :] = self.no2_total[i, no2_nline - 2, :]
            self.no2_strat[i, :, :] = self.no2_total[i, :, :] - self.no2_tropo[i, :, :]
        self.no2_strat[self.no2_strat < 0.0] = 0.0
        self.no2_total = self.no2_total * 1.0e15
        self.no2_tropo = self.no2_tropo * 1.0e15
        self.no2_strat = self.no2_strat * 1.0e15

    def download(self):
        #        t=time.time()
        ancurl_prefix = (
            "https://oceandata.sci.gsfc.nasa.gov/ob/getfile/"  # NASA ancillary data
        )
        oz_postfix = "_O3_AURAOMI_24h.hdf"
        oz_postfix2 = "_O3_EPTOMS_24h.hdf"
        # met_postfix = "_MET_NCEPR2_6h.hdf"
        met_postfix = "_MET_NCEP_6h.hdf"
        server = "oceandata.sci.gsfc.nasa.gov"
        self.l1btime = datetime.strptime(self.datestr, "%Y%m%d%H%M%S").timestamp()
        self.doy = datetime.strptime(self.datestr, "%Y%m%d%H%M%S").timetuple().tm_yday
        if int(self.datestr[8:14]) < 120000:  # before mid day
            dt = (
                datetime.strptime(self.datestr[0:8] + "000000", "%Y%m%d%H%M%S")
                - timedelta(days=1)
            ).timetuple()
            self.oz1_name = (
                "N" + str(dt.tm_year) + "{:03d}".format(dt.tm_yday) + "00" + oz_postfix
            )
            self.oz2_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "00" + oz_postfix
            )
            self.ozdt = (
                (
                    self.l1btime
                    - (
                        datetime.strptime(
                            self.datestr[0:8] + "000000", "%Y%m%d%H%M%S"
                        ).timestamp()
                        - 12 * 3600
                    )
                )
                / 3600
                / 24
            )
        else:
            dt = (
                datetime.strptime(self.datestr[0:8] + "000000", "%Y%m%d%H%M%S")
                + timedelta(days=1)
            ).timetuple()
            self.oz1_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "00" + oz_postfix
            )
            self.oz2_name = (
                "N" + str(dt.tm_year) + "{:03d}".format(dt.tm_yday) + "00" + oz_postfix
            )
            self.ozdt = (
                (
                    self.l1btime
                    - datetime.strptime(
                        self.datestr[0:8] + "120000", "%Y%m%d%H%M%S"
                    ).timestamp()
                )
                / 3600
                / 24
            )

        if int(self.datestr[8:14]) <= 60000:  # before 06:00
            self.met1_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "00" + met_postfix
            )
            self.met2_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "06" + met_postfix
            )
            self.metdt = (
                (
                    self.l1btime
                    - datetime.strptime(
                        self.datestr[0:8] + "000000", "%Y%m%d%H%M%S"
                    ).timestamp()
                )
                / 3600
                / 24
                / 0.25
            )
        elif int(self.datestr[8:14]) <= 120000:  # before 12:00
            self.met1_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "06" + met_postfix
            )
            self.met2_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "12" + met_postfix
            )
            self.metdt = (
                (
                    self.l1btime
                    - datetime.strptime(
                        self.datestr[0:8] + "060000", "%Y%m%d%H%M%S"
                    ).timestamp()
                )
                / 3600
                / 24
                / 0.25
            )
        elif int(self.datestr[8:14]) <= 180000:  # before 18:00
            self.met1_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "12" + met_postfix
            )
            self.met2_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "18" + met_postfix
            )
            self.metdt = (
                (
                    self.l1btime
                    - datetime.strptime(
                        self.datestr[0:8] + "120000", "%Y%m%d%H%M%S"
                    ).timestamp()
                )
                / 3600
                / 24
                / 0.25
            )
        else:  # after 18:00
            dt = (
                datetime.strptime(self.datestr[0:8] + "000000", "%Y%m%d%H%M%S")
                + timedelta(days=1)
            ).timetuple()
            self.met1_name = (
                "N" + self.datestr[0:4] + "{:03d}".format(self.doy) + "18" + met_postfix
            )
            self.met2_name = (
                "N" + str(dt.tm_year) + "{:03d}".format(dt.tm_yday) + "00" + met_postfix
            )
            self.metdt = (
                (
                    self.l1btime
                    - datetime.strptime(
                        self.datestr[0:8] + "180000", "%Y%m%d%H%M%S"
                    ).timestamp()
                )
                / 3600
                / 24
                / 0.25
            )

        if exists(self.path + self.oz1_name):
            print("Ozone file {} located on local drive.".format(self.oz1_name))
        elif exists(self.path + self.oz1_name[0:10] + oz_postfix2):
            self.oz1_name = self.oz1_name[0:10] + oz_postfix2
            print("Ozone file {} located on local drive.".format(self.oz1_name))
        else:
            print(
                "Ozone file {} not found on local drive, downloading from NASA OBPG ...".format(
                    self.oz1_name
                )
            )
            request = "/ob/getfile/" + self.oz1_name
            status = httpdl(server, request, localpath=self.path, uncompress=False)
            if status:
                print("OMI ozone data unavailable, downloading TOMS ozone data ...")
                self.oz1_name = self.oz1_name[0:10] + oz_postfix2  # rename oz file
                request = "/ob/getfile/" + self.oz1_name
                status = httpdl(server, request, localpath=self.path, uncompress=False)
                if status:
                    print(
                        "Real time ozone data unavailable, using climatology data ..."
                    )
                    self.oz1_name = "ozone_climatology_v2014.hdf"  # rename oz file to the climatology data

        if exists(self.path + self.oz2_name):
            print("Ozone file {} located on local drive.".format(self.oz2_name))
        elif exists(self.path + self.oz2_name[0:10] + oz_postfix2):
            self.oz2_name = self.oz2_name[0:10] + oz_postfix2
            print("Ozone file {} located on local drive.".format(self.oz2_name))
        else:
            print(
                "Ozone file {} not found on local drive, downloading from NASA OBPG ...".format(
                    self.oz2_name
                )
            )
            request = "/ob/getfile/" + self.oz2_name
            status = httpdl(server, request, localpath=self.path, uncompress=False)
            if status:
                print("OMI ozone data unavailable, downloading TOMS ozone data ...")
                self.oz2_name = self.oz2_name[0:10] + oz_postfix2  # rename oz file
                request = "/ob/getfile/" + self.oz2_name
                status = httpdl(server, request, localpath=self.path, uncompress=False)
                if status:
                    print(
                        "Real time ozone data unavailable, using climatology data ..."
                    )
                    self.oz2_name = "ozone_climatology_v2014.hdf"  # rename oz file to the climatology data

        if exists(self.path + self.met1_name):
            print("MET file {} located on local drive.".format(self.met1_name))
        else:
            print(
                "MET file {} not found on local drive, downloading from NASA OBPG ...".format(
                    self.met1_name
                )
            )
            request = "/ob/getfile/" + self.met1_name
            status = httpdl(server, request, localpath=self.path, uncompress=False)
            if status:
                print("Real time MET data unavailable ...")

        if exists(self.path + self.met2_name):
            print("MET file {} located on local drive.".format(self.met2_name))
        else:
            print(
                "MET file {} not found on local drive, downloading from NASA OBPG ...".format(
                    self.met2_name
                )
            )
            request = "/ob/getfile/" + self.met2_name
            status = httpdl(server, request, localpath=self.path, uncompress=False)
            if status:
                print("Real time MET data unavailable ...")

    def read_ozone(self):
        print("Reading Ozone data ...")
        self.oz_lat = np.arange(90.5, -91.5, -1)
        self.oz_lon = np.arange(-180.5, 181.5, 1)
        oz_nline = len(self.oz_lat)
        oz_npixl = len(self.oz_lon)
        if "climatology" in self.oz1_name or "climatology" in self.oz2_name:
            print(
                "Warning: real time ozone data unavailable, using climatology ozone data ..."
            )
        if "climatology" in self.oz1_name:
            f1 = SD("./auxdata/common/" + self.oz1_name, SDC.READ)
            ozone1 = f1.select("ozone_mean_" + "{:03d}".format(self.doy))[:, :] * 0.001
        elif "OMI" in self.oz1_name:
            f1 = SD(self.path + self.oz1_name, SDC.READ)
            ozone1 = f1.select("ozone")[:, :] * 0.001
        elif "TOMS" in self.oz1_name:
            f1 = SD(self.path + self.oz1_name, SDC.READ)
            data = f1.select("ozone")[:, :] * 0.001
            # TOMS ozone data use a different grid, interpolate to 1 degree grid point
            lat = np.arange(89.5, -90.5, -1)
            lon = np.arange(-180.625, 181.875, 1.25)
            nline = len(lat)
            npix = len(lon)
            ozone = np.zeros((nline, npix), dtype="float64")
            ozone[:, 1 : npix - 1] = data
            ozone[:, 0] = data[:, -1]
            ozone[:, -1] = data[:, 0]
            grid_yt = np.arange(-179.5, 180.5, 1)
            func = interpolate.interp2d(lon, lat, ozone, kind="linear")
            ozone1 = np.flip(np.flip(func(grid_yt, lat), 1))

        if "climatology" in self.oz2_name:
            f2 = SD("./auxdata/common/" + self.oz2_name, SDC.READ)
            ozone2 = f2.select("ozone_mean_" + "{:03d}".format(self.doy))[:, :] * 0.001
        elif "OMI" in self.oz2_name:
            f2 = SD(self.path + self.oz2_name, SDC.READ)
            ozone2 = f2.select("ozone")[:, :] * 0.001
        elif "TOMS" in self.oz2_name:
            f2 = SD(self.path + self.oz2_name, SDC.READ)
            data = f2.select("ozone")[:, :] * 0.001
            # TOMS ozone data use a different grid, interpolate to 1 degree grid point
            lat = np.arange(89.5, -90.5, -1)
            lon = np.arange(-180.625, 181.875, 1.25)
            nline = len(lat)
            npix = len(lon)
            ozone = np.zeros((nline, npix), dtype="float64")
            ozone[:, 1 : npix - 1] = data
            ozone[:, 0] = data[:, -1]
            ozone[:, -1] = data[:, 0]
            grid_yt = np.arange(-179.5, 180.5, 1)
            func = interpolate.interp2d(lon, lat, ozone, kind="linear")
            ozone2 = np.flip(np.flip(func(grid_yt, lat), 1))

        ozone = (
            ozone1 * (1 - self.ozdt) + ozone2 * self.ozdt
        )  # interpolate in time and convert unit
        self.ozmap = np.zeros((oz_nline, oz_npixl), dtype="float64")
        self.ozmap[1 : oz_nline - 1, 1 : oz_npixl - 1] = ozone
        self.ozmap[:, 0] = self.ozmap[:, oz_npixl - 2]
        self.ozmap[:, oz_npixl - 1] = self.ozmap[:, 1]
        self.ozmap[0, :] = self.ozmap[1, :]
        self.ozmap[oz_nline - 1, :] = self.ozmap[oz_nline - 2, :]

    def read_met(self):
        print("Reading windspeed, pressure and RH data ...")
        f1 = SD(self.path + self.met1_name, SDC.READ)
        zwind = f1.select("z_wind")[:, :]
        mwind = f1.select("m_wind")[:, :]
        ws1 = np.power(np.power(zwind, 2) + np.power(mwind, 2), 0.5)
        press1 = f1.select("press")[:, :]
        rh1 = f1.select("rel_hum")[:, :]

        f2 = SD(self.path + self.met2_name, SDC.READ)
        zwind = f2.select("z_wind")[:, :]
        mwind = f2.select("m_wind")[:, :]
        ws2 = np.power(np.power(zwind, 2) + np.power(mwind, 2), 0.5)
        press2 = f2.select("press")[:, :]
        rh2 = f2.select("rel_hum")[:, :]

        ws = ws1 * (1 - self.metdt) + ws2 * self.metdt
        press = press1 * (1 - self.metdt) + press2 * self.metdt
        rh = rh1 * (1 - self.metdt) + rh2 * self.metdt

        self.met_lat = np.arange(91, -92, -1)
        self.met_lon = np.arange(-180.5, 181.5, 1)
        met_nline = len(self.met_lat)
        met_npixl = len(self.met_lon)
        self.wsmap = np.zeros((met_nline, met_npixl), dtype="float64")
        self.pressmap = np.zeros((met_nline, met_npixl), dtype="float64")
        self.rhmap = np.zeros((met_nline, met_npixl), dtype="float64")

        self.wsmap[1 : met_nline - 1, 1 : met_npixl - 1] = ws
        self.wsmap[:, 0] = self.wsmap[:, met_npixl - 2]
        self.wsmap[:, met_npixl - 1] = self.wsmap[:, 1]
        self.wsmap[0, :] = self.wsmap[1, :]
        self.wsmap[met_nline - 1, :] = self.wsmap[met_nline - 2, :]

        self.pressmap[1 : met_nline - 1, 1 : met_npixl - 1] = press
        self.pressmap[:, 0] = self.pressmap[:, met_npixl - 2]
        self.pressmap[:, met_npixl - 1] = self.pressmap[:, 1]
        self.pressmap[0, :] = self.pressmap[1, :]
        self.pressmap[met_nline - 1, :] = self.pressmap[met_nline - 2, :]

        self.rhmap[1 : met_nline - 1, 1 : met_npixl - 1] = rh
        self.rhmap[:, 0] = self.rhmap[:, met_npixl - 2]
        self.rhmap[:, met_npixl - 1] = self.rhmap[:, 1]
        self.rhmap[0, :] = self.rhmap[1, :]
        self.rhmap[met_nline - 1, :] = self.rhmap[met_nline - 2, :]

    def trans_ozone(self, k_oz, l1b_lat, l1b_lon, l1b_solz, l1b_senz):
        # interpolate ozone map to the L1B grid and compute transmittance
        print("Compute Ozone transmittance ...")
        # npix=len(l1b_lat)
        func = interpolate.RegularGridInterpolator(
            (np.flip(self.oz_lat), self.oz_lon), np.flip(self.ozmap, 0)
        )
        l1b_oz = func(np.array([l1b_lat, l1b_lon]).transpose())
        self.l1b_oz = l1b_oz
        tg_sol = np.exp(
            np.matmul(np.expand_dims(-l1b_oz / np.cos(np.deg2rad(l1b_solz)), 1), [k_oz])
        )
        tg_sen = np.exp(
            np.matmul(np.expand_dims(-l1b_oz / np.cos(np.deg2rad(l1b_senz)), 1), [k_oz])
        )
        return tg_sol, tg_sen

    def trans_no2(self, k_no2, month, l1b_lat, l1b_lon, l1b_solz, l1b_senz):
        # interpolate no2 map to the L1B grid and compute transmittance
        print("Compute NO2 transmittance ...")
        # npix=len(l1b_lat)
        a_285 = k_no2 * (1.0 - 0.003 * (285.0 - 294.0))
        a_225 = k_no2 * (1.0 - 0.003 * (225.0 - 294.0))
        func = interpolate.RegularGridInterpolator(
            (np.flip(self.no2_frac_lat), self.no2_frac_lon), np.flip(self.no2_frac, 0)
        )
        l1b_no2_frac = func(np.array([l1b_lat, l1b_lon]).transpose())
        no2_strat = self.no2_strat[int(month) - 1, :, :]
        func = interpolate.RegularGridInterpolator(
            (np.flip(self.no2_lat), self.no2_lon), np.flip(no2_strat, 0)
        )
        l1b_no2_strat = func(np.array([l1b_lat, l1b_lon]).transpose())
        no2_tropo = self.no2_tropo[int(month) - 1, :, :]
        func = interpolate.RegularGridInterpolator(
            (np.flip(self.no2_lat), self.no2_lon), np.flip(no2_tropo, 0)
        )
        l1b_no2_tropo = func(np.array([l1b_lat, l1b_lon]).transpose())
        l1b_no2_trop200 = l1b_no2_frac * l1b_no2_tropo
        l1b_no2_trop200[l1b_no2_trop200 < 0] = 0.0
        tg_sol = np.exp(
            -(
                np.matmul(
                    np.expand_dims(l1b_no2_trop200 / np.cos(np.deg2rad(l1b_solz)), 1),
                    [a_285],
                )
                + np.matmul(
                    np.expand_dims(l1b_no2_strat / np.cos(np.deg2rad(l1b_solz)), 1),
                    [a_225],
                )
            )
        )
        tg_sen = np.exp(
            -(
                np.matmul(
                    np.expand_dims(l1b_no2_trop200 / np.cos(np.deg2rad(l1b_senz)), 1),
                    [a_285],
                )
                + np.matmul(
                    np.expand_dims(l1b_no2_strat / np.cos(np.deg2rad(l1b_senz)), 1),
                    [a_225],
                )
            )
        )
        self.l1b_no2_frac = l1b_no2_frac
        self.l1b_no2_strat = l1b_no2_strat
        self.l1b_no2_tropo = l1b_no2_tropo
        return tg_sol, tg_sen

    def trans_o2_aer(self, l1b_solz, l1b_senz):
        # O2 transmittance for aerosols
        ao2 = np.array([-1.0796, 9.0481e-2, -6.8452e-3])
        airmass = 1 / np.cos(np.deg2rad(l1b_solz)) + 1 / np.cos(np.deg2rad(l1b_senz))
        t_o2 = 1.0 + np.power(10, ao2[0] + ao2[1] * airmass + ao2[2] * airmass**2)
        return t_o2

    def trans_o2_ray(self, l1b_solz, l1b_senz):
        # O2 transmittance for Rayleigh
        ao2 = np.array([-1.3491, 0.1155, -7.0218e-3])
        airmass = 1 / np.cos(np.deg2rad(l1b_solz)) + 1 / np.cos(np.deg2rad(l1b_senz))
        ray_o2 = 1.0 / (
            1.0 + np.power(10, ao2[0] + ao2[1] * airmass + ao2[2] * airmass**2)
        )
        return ray_o2

    def get_metdata(self, l1b_lat, l1b_lon):
        # interpolate pressure, relative humidity (RH) and windspeed to the L1B grid
        func = interpolate.RegularGridInterpolator(
            (np.flip(self.met_lat), self.met_lon), np.flip(self.pressmap, 0)
        )
        l1b_press = func(np.array([l1b_lat, l1b_lon]).transpose())

        func = interpolate.RegularGridInterpolator(
            (np.flip(self.met_lat), self.met_lon), np.flip(self.rhmap, 0)
        )
        l1b_rh = func(np.array([l1b_lat, l1b_lon]).transpose())

        func = interpolate.RegularGridInterpolator(
            (np.flip(self.met_lat), self.met_lon), np.flip(self.wsmap, 0)
        )
        l1b_ws = func(np.array([l1b_lat, l1b_lon]).transpose())

        return l1b_press, l1b_rh, l1b_ws

    def whitecaps(self, band, l1b_solz, l1b_senz, l1b_ws, l1b_pressure, taur):
        print("Compute whitecaps reflectance ...")
        awc_band = np.array([380, 412, 443, 490, 510, 555, 670, 765, 865, 1000, 5000])
        awc_tab = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.889225, 0.760046, 0.644950, 0.0, 0.0]
        )
        wc_ws_min = 6.33
        wc_ws_max = 12
        p0 = 1013.25
        npix = len(l1b_ws)
        nband = len(band)
        l1b_tlf = np.zeros([npix, nband])
        func = interpolate.interp1d(awc_band, awc_tab, kind="linear")
        awc = func([band])
        idx1 = l1b_ws <= wc_ws_max
        idx2 = l1b_ws >= wc_ws_min
        idx = idx1 & idx2
        x = np.matmul(
            np.expand_dims(1.925e-5 * np.power(l1b_ws[idx] - wc_ws_min, 3), 1), [awc]
        )
        l1b_tlf[idx, :] = x
        tg_sol = np.exp(
            -0.5
            * np.matmul(
                np.expand_dims(l1b_pressure / p0 / np.cos(np.deg2rad(l1b_solz)), 1),
                [taur],
            )
        )
        tg_sen = np.exp(
            -0.5
            * np.matmul(
                np.expand_dims(l1b_pressure / p0 / np.cos(np.deg2rad(l1b_senz)), 1),
                [taur],
            )
        )
        x1 = np.zeros([npix, nband])
        for i in range(nband):
            x1[:, i] = np.cos(np.deg2rad(l1b_solz))
        l1b_tlf = l1b_tlf / np.pi * x1 * tg_sol * tg_sen
        return l1b_tlf
