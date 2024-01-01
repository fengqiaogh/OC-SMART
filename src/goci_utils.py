#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:08:04 2019

@author: Ryan Chang
"""

import numpy as np

PARAM_INDEX_A = 5
PARAM_INDEX_B = 14
PARAM_INDEX_C = 22
PARAM_INDEX_D = 31

def goci_slots(nav_data, height, width, band):
    di = np.full([height, width, 16], float('-inf'))

    for slot_index in range(16):
        y, x = np.mgrid[0:height, 0:width]
        sbp = nav_data[band * 16 + slot_index]
        sbp = np.hstack((sbp[4], sbp[5], sbp[6], sbp[7], sbp[13][:9], sbp[15][:8], sbp[17][:9], sbp[19][:8]))

        x = (x - sbp[0]) / sbp[2]
        y = (y - sbp[1]) / sbp[3]

        xs = np.power(x, 2)
        ys = np.power(y, 2)

        matrix = np.array([x, y, np.multiply(x, y), xs, ys, np.multiply(xs, y),
                           np.multiply(x, ys), np.multiply(xs, ys)])

        del x, y, xs, ys

        P1 = np.full((height, width), sbp[PARAM_INDEX_A - 1])
        P3 = np.full((height, width), sbp[PARAM_INDEX_C - 1])
        P2 = np.full((height, width), 1.0)
        P4 = np.full((height, width), 1.0)

        for param_index in range(8):
            P1 += matrix[param_index] * sbp[PARAM_INDEX_A + param_index]
            P3 += matrix[param_index] * sbp[PARAM_INDEX_C + param_index]
            P2 += matrix[param_index] * sbp[PARAM_INDEX_B + param_index - 1]
            P4 += matrix[param_index] * sbp[PARAM_INDEX_D + param_index - 1]

        del matrix, sbp

        xn = P1 / P2
        yn = P3 / P4

        del P1, P2, P3, P4

        valid = (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
        di[valid, slot_index] = np.minimum(np.minimum(1.0 - xn[valid], 1.0 + xn[valid]),
                                           np.minimum(1.0 - yn[valid], 1.0 + yn[valid]))

        print("\tFinished slot", slot_index)

    slots = np.argmax(di, axis = 2) + 1
    slots[np.amax(di, axis = 2) == float('-inf')] = -1
    return slots

def goci_slots_time(nav_data):
    rel_time = np.zeros(16)
    for slot_index in range(16):
        for band_index in range(8):
            rel_time[slot_index] += nav_data[band_index * 16 + slot_index][2]

    rel_time /= 8
    return rel_time

def goci_solar_angles(year, month, day, hour, minute, second, latitude, longitude):
    # Calculate difference in days between the current Julian Day and JD 2451545.0, which is noon 1 January 2000 Universal Time
    EARTH_RADIUS_KM = 6371.0
    ASTRONOMICAL_UNIT = 1.4959787e8
    # Calculate time of the day in UT decimal hours
    dDecimalHours = hour + minute / 60.0 + second / 3600.0
    # Calculate current Julian Day
    liAux1 = (month - 14) // 12
    liAux2 = (1461 * (year + 4800 + liAux1)) // 4 + (367 * (month - 2 - 12 * liAux1)) // 12 - (
                3 * ((year + 4900 + liAux1) // 100)) // 4 + day - 32075
    dJulianDate = liAux2 - 0.5 + dDecimalHours / 24

    # Calculate difference between current Julian Day and JD 2451545.0
    dElapsedJulianDays = dJulianDate - 2451545.0

    # Calculate ecliptic coordinates (ecliptic longitude and obliquity of the ecliptic in radians but without limiting the angle to be less than 2*Pi (i.e., the result may be greater than 2*Pi)
    dOmega = 2.1429 - 0.0010394594 * dElapsedJulianDays
    dMeanLongitude = 4.8950630 + 0.017202791698 * dElapsedJulianDays  # in radians
    dMeanAnomaly = 6.2400600 + 0.0172019699 * dElapsedJulianDays
    dEclipticLongitude = dMeanLongitude + 0.03341607 * np.sin(dMeanAnomaly) + 0.00034894 * np.sin(
        2 * dMeanAnomaly) - 0.0001134 - 0.0000203 * np.sin(dOmega)
    dEclipticObliquity = 0.4090928 - 6.2140e-9 * dElapsedJulianDays + 0.0000396 * np.cos(dOmega)

    # Calculate celestial coordinates ( RA and DEC ) in radians but without limiting the angle to be less than 2*Pi (i.e., the result may be greater than 2*Pi)
    dSin_EclipticLongitude = np.sin(dEclipticLongitude)
    dY = np.cos(dEclipticObliquity) * dSin_EclipticLongitude
    dX = np.cos(dEclipticLongitude)
    dRightAscension = np.arctan2(dY, dX)
    if dRightAscension < 0.0:
        dRightAscension = dRightAscension + 2.0 * np.pi
    dDeclination = np.arcsin(np.sin(dEclipticObliquity) * dSin_EclipticLongitude)

    # Calculate local coordinates ( azimuth and zenith angle ) in degrees
    dGreenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * dElapsedJulianDays + dDecimalHours
    dLocalMeanSiderealTime = np.deg2rad(dGreenwichMeanSiderealTime * 15 + longitude)
    dHourAngle = dLocalMeanSiderealTime - dRightAscension
    dLatitudeInRadians = np.deg2rad(latitude)
    dCos_Latitude = np.cos(dLatitudeInRadians)
    dSin_Latitude = np.sin(dLatitudeInRadians)
    dCos_HourAngle = np.cos(dHourAngle)
    m_solar_zenith_angle_degree = np.arccos(
        dCos_Latitude * dCos_HourAngle * np.cos(dDeclination) + np.sin(dDeclination) * dSin_Latitude)
    dY = -1 * np.sin(dHourAngle)
    dX = np.tan(dDeclination) * dCos_Latitude - dSin_Latitude * dCos_HourAngle
    m_solar_azimuth_angle_degree = np.arctan2(dY, dX)
    m_solar_azimuth_angle_degree[m_solar_azimuth_angle_degree < 0.0] = m_solar_azimuth_angle_degree[m_solar_azimuth_angle_degree < 0.0] + 2.0 * np.pi
    m_solar_azimuth_angle_degree = np.rad2deg(m_solar_azimuth_angle_degree)
    # Parallax Correction
    dParallax = (EARTH_RADIUS_KM / ASTRONOMICAL_UNIT) * np.sin(m_solar_zenith_angle_degree)
    m_solar_zenith_angle_degree = np.rad2deg(m_solar_zenith_angle_degree + dParallax)
    return m_solar_zenith_angle_degree, m_solar_azimuth_angle_degree


def goci_sensor_zenith(latitude, longitude):
    GEOSTATIONARY_ORBIT_ALTITUDE_KM = 35786.0
    EARTH_RADIUS_KM = 6371.0
    GOCI_ORBIT_LONGITUDE = 128.2
    longdiffr = np.deg2rad(longitude - GOCI_ORBIT_LONGITUDE)
    eslatr = np.deg2rad(latitude)
    r1 = 1.0 + GEOSTATIONARY_ORBIT_ALTITUDE_KM / EARTH_RADIUS_KM
    v1 = r1 * np.cos(eslatr) * np.cos(longdiffr) - 1.0
    v2 = r1 * np.sqrt(1 - np.cos(eslatr) * np.cos(eslatr) * np.cos(longdiffr) * np.cos(longdiffr))
    eselevation = np.rad2deg(np.arctan(v1 / v2))
    eselrefracted = eselevation
    eselrefracted[eselevation < 30.0] = (eselevation[eselevation < 30.0] + np.sqrt(
        np.power(eselevation[eselevation < 30.0], 2) + 4.132)) / 2.0
    m_satellite_zenith_angle_degree = 90.0 - eselrefracted
    return m_satellite_zenith_angle_degree


def goci_sensor_azimuth(latitude, longitude, satellite_longitude):
    longdiffr_rad = np.deg2rad(longitude - satellite_longitude)
    esazimuth_deg = 180.0 + np.rad2deg(np.arctan(np.tan(longdiffr_rad) / np.sin(np.deg2rad(latitude))))
    esazimuth_deg[latitude < 0.0] = esazimuth_deg[latitude < 0.0] - 180.0
    esazimuth_deg[esazimuth_deg < 0.0] = esazimuth_deg[esazimuth_deg < 0.0] + 360.0
    return esazimuth_deg
