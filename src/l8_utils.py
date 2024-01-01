# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:25:19 2019

@author: Yongzhen
"""

import json
import re

######### Functions to read landsat8 MTL file ##############################
def load_mtl(src_mtl):
    with open(src_mtl) as src:
        if src_mtl.split('.')[-1] == 'json':
            return json.loads(src.read())
        else:
            return parse_mtl_txt(src.read())
def parse_mtl_txt(mtltxt):
    group = re.findall('.*\n', mtltxt)

    is_group = re.compile(r'GROUP\s\=\s.*')
    is_end = re.compile(r'END_GROUP\s\=\s.*')
    get_group = re.compile('\=\s([A-Z0-9\_]+)')

    output = [{
            'key': 'all',
            'data': {}
        }]

    for g in map(str.lstrip, group):
        if is_group.match(g):
            output.append({
                    'key': get_group.findall(g)[0],
                    'data': {}
                })

        elif is_end.match(g):
            endk = output.pop()
            k = u'{}'.format(endk['key'])
            output[-1]['data'][k] = endk['data']

        else:
            k, d = _parse_data(g)
            if k:
                k = u'{}'.format(k)
                output[-1]['data'][k] = d

    return output[0]['data']

def _cast_to_best_type(kd):
    key, data = kd[0]
    try:
        return key, int(data)
    except ValueError:
        try:
            return key, float(data)
        except ValueError:
            return key, u'{}'.format(data.strip('"'))


def _parse_data(line):
    kd = re.findall(r'(.*)\s\=\s(.*)', line)

    if len(kd) == 0:
        return False, False
    else:
        return _cast_to_best_type(kd)  
########################################################################      