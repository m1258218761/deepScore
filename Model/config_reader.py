# -*- coding:utf-8 -*-
# author:mx
# datetime:2020/4/25 15:06
# email:minxinm@foxmail.com
import configparser


class MyParser(configparser.ConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d
