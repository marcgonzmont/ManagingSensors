# from os.path import isfile, join, altsep
# from os import listdir
# from numba import jit
#
# @jit
# def getSamples(path, ext=''):
#     '''
#     Auxiliary function that extracts file names from a given path based on extension
#     :param path: source path
#     :param ext: file extension
#     :return: array with samples
#     '''
#     samples = [altsep.join((path, f)) for f in listdir(path)
#               if isfile(join(path, f)) and f.endswith(ext)]
#
#     if len(samples) == 0:
#         print("ERROR!!! ARRAY OF SAMPLES IS EMPTY (check file extension)")
#
#     return samples
