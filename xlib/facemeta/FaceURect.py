import operator
from collections import Iterable
from typing import List, Tuple

import cv2
import numpy as np
import numpy.linalg as npla
from .. import math as lib_math
from ..math import Affine2DMat, Affine2DUniMat


class FaceURect:
    """
    Describes face rectangle in uniform float coordinates
    """
    def __init__(self):
        self._pts = None

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)

    @staticmethod
    def sort_by_area_size(rects : List['FaceURect']):
        """
        sort list of FaceURect by largest area descend
        """
        rects = [ (rect, rect.get_area()) for rect in rects ]
        rects = sorted(rects, key=operator.itemgetter(1), reverse=True )
        rects = [ x[0] for x in rects]
        return rects

    @staticmethod
    def sort_by_dist_from_center(rects : List['FaceURect']):
        """
        sort list of FaceURect by nearest distance from center to center of rects descent
        """
        c = np.float32([0.5,0.5])

        rects = [ (rect, npla.norm( rect.get_center_point()-c )) for rect in rects ]
        rects = sorted(rects, key=operator.itemgetter(1) )
        rects = [ x[0] for x in rects]
        return rects

    @staticmethod
    def from_4pts(pts : Iterable):
        """
        Construct FaceURect from 4 pts
         0--3
         |  |
         1--2
        """
        if not isinstance(pts, Iterable):
            raise ValueError('pts must be Iterable')

        pts = np.array(pts, np.float32)
        if pts.shape != (4,2):
            raise ValueError('pts must have (4,2) shape')

        face_rect = FaceURect()
        face_rect._pts = pts
        return face_rect

    @staticmethod
    def from_ltrb(ltrb : Iterable):
        """
        Construct FaceURect from l,t,r,b list of float values
           t
         l-|-r
           b
        """
        if not isinstance(ltrb, Iterable):
            raise ValueError('ltrb must be Iterable')

        l,t,r,b = ltrb
        return FaceURect.from_4pts([ [l,t], [l,b], [r,b], [r,t] ])


    def get_area(self, w_h = None) -> float:
        """
        get area of rectangle.

         w_h(None)    provide (w,h) to scale uniform rect to target size
        """
        return lib_math.polygon_area(self.as_4pts(w_h))

    def get_center_point(self, w_h = None) -> np.ndarray:
        """

         w_h(None)    provide (w,h) to scale uniform rect to target size

        returns np.ndarray (2,)
        """
        pts = self.as_4pts(w_h)
        return np.mean(pts, 0)

    def as_ltrb_bbox(self, w_h = None) -> np.ndarray:
        """
        get bounding box of rect as left,top,right,bottom

         w_h(None)    provide (w,h) to scale uniform rect to target size

        returns np.ndarray with l,t,r,b values
        """
        pts = self.as_4pts( w_h=w_h)
        return np.array( [np.min(pts[:,0]), np.max(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,1])], np.float32 )

    def as_4pts(self, w_h = None) -> np.ndarray:
        """
        get rect as 4 pts

         0--3
         |  |
         1--2

         w_h(None)    provide (w,h) to scale uniform rect to target size

        returns np.ndarray (4,2) 4 pts with w,h
        """
        if w_h is not None:
            return self._pts * w_h
        return self._pts.copy()

    def transform(self, mat, invert=False) -> 'FaceURect':
        """
        Tranforms FaceURect using affine mat and returns new FaceURect()

         mat : np.ndarray   should be uniform affine mat
        """

        if not isinstance(mat, np.ndarray):
            raise ValueError('mat must be an instance of np.ndarray')

        if invert:
            mat = cv2.invertAffineTransform (mat)

        pts = self._pts.copy()
        pts = np.expand_dims(pts, axis=1)
        pts = cv2.transform(pts, mat, pts.shape).squeeze()

        return FaceURect.from_4pts(pts)

    def cut(self, img : np.ndarray, coverage : float, output_size : int) -> Tuple[Affine2DMat, Affine2DUniMat]:
        """
        Cut the face to square of output_size from img with given coverage using this rect

        returns image,
                uni_mat     uniform matrix to transform uniform img space to uniform cutted space
        """

        # Face rect is not a square, also rect can be rotated

        h,w = img.shape[0:2]

        # Get scaled rect pts to target img
        pts = self.as_4pts( w_h=(w,h) )

        # Estimate transform from global space to local aligned space with bounds [0..1]
        mat = Affine2DMat.umeyama(pts, uni_rect, True)

        # get corner points in global space
        g_p = mat.invert().transform_points ( [(0,0),(0,1),(1,1),(1,0),(0.5,0.5)] )
        g_c = g_p[4]

        # calc diagonal vectors between corners in global space
        tb_diag_vec = lib_math.segment_to_vector(g_p[0], g_p[2]).astype(np.float32)
        bt_diag_vec = lib_math.segment_to_vector(g_p[1], g_p[3]).astype(np.float32)

        mod = lib_math.segment_length(g_p[0],g_p[4])*coverage

        l_t = np.array( [ g_c - tb_diag_vec*mod,
                          g_c + bt_diag_vec*mod,
                          g_c + tb_diag_vec*mod ], np.float32 )

        mat     = Affine2DMat.from_3_pairs ( l_t, np.float32(( (0,0),(output_size,0),(output_size,output_size) )))
        uni_mat = Affine2DUniMat.from_3_pairs ( (l_t/(w,h)).astype(np.float32), np.float32(( (0,0),(1,0),(1,1) )) )

        face_image = cv2.warpAffine(img, mat, (output_size, output_size), cv2.INTER_CUBIC )
        return face_image, uni_mat


    def draw(self, img : np.ndarray, color, thickness=1):
        """
        draw rect on the img scaled by img.wh

         color  tuple of values      should be the same as img color channels
        """
        h,w = img.shape[0:2]
        pts = self.as_4pts(w_h=(w,h)).astype(np.int32)
        pts_len = len(pts)
        for i in range (pts_len):
            p0 = tuple( pts[i] )
            p1 = tuple( pts[ (i+1) % pts_len] )
            cv2.line (img, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA)

uni_rect = np.array([
[0.0, 0.0],
[0.0, 1.0],
[1.0, 1.0],
[1.0, 0.0],
], dtype=np.float32)
