from typing import Tuple
import numpy as np
from .. import math as lib_math


class FacePose:
    """
    Describes face pitch/yaw/roll
    """
    def __init__(self):
        self._pyr : np.ndarray = None

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)

    def as_radians(self) -> Tuple[float, float, float]:
        """
        returns pitch,yaw,roll in radians
        """
        return self._pyr.copy()

    def as_degress(self) -> Tuple[float, float, float]:
        """
        returns pitch,yaw,roll in degrees
        """
        return np.degrees(self._pyr)

    @staticmethod
    def from_radians(pitch, yaw, roll):
        """
        """
        face_rect = FacePose()
        face_rect._pyr = np.array([pitch, yaw, roll], np.float32)
        return face_rect

    @staticmethod
    def from_3D_468_landmarks(lmrks):
        """
        """
        mat = np.empty((3,3))
        mat[0,:] = (lmrks[454] - lmrks[234])/np.linalg.norm(lmrks[454] - lmrks[234])
        mat[1,:] = (lmrks[152] - lmrks[6])/np.linalg.norm(lmrks[152] - lmrks[6])
        mat[2,:] = np.cross(mat[0, :], mat[1, :])
        pitch, yaw, roll = lib_math.rotation_matrix_to_euler(mat)

        return FacePose.from_radians(pitch, yaw, roll)
