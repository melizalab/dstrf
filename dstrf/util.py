# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""utility functions and classes"""
import argparse
import json


def assoc_in(dct, path, value):
    for x in path:
        prev, dct = dct, dct.setdefault(x, {})
    prev[x] = value


class ParseKeyVal(argparse.Action):

    def __call__(self, parser, namespace, arg, option_string=None):
        kv = getattr(namespace, self.dest)
        if kv is None:
            kv = dict()
        if not arg.count('=') == 1:
            raise ValueError(
                "-k %s argument badly formed; needs key=value" % arg)
        else:
            key, val = arg.split('=')
            try:
                kv[key] = json.loads(val)
            except json.decoder.JSONDecodeError:
                kv[key] = val
        setattr(namespace, self.dest, kv)
