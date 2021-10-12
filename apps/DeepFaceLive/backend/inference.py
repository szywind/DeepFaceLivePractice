import cv2
import os
import yaml
from pathlib import Path
import numpy as np
from enum import IntEnum
import numexpr as ne

from modelhub import DFLive
from modelhub import onnx as onnx_models
from modelhub import cv as cv_models

from xlib import os as lib_os
from xlib.facemeta import FaceMark, FaceURect
from xlib.facemeta import FaceULandmarks, FacePose
from xlib.facemeta import FaceAlign
from xlib.facemeta import FaceMask, FaceSwap

from xlib.image import ImageProcessor
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None

from xlib.onnxruntime.device import ORTDeviceInfo, get_available_devices_info, get_cpu_device

def read_image(filepath):
    with open(filepath, "rb") as stream:
        bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, flags=cv2.IMREAD_UNCHANGED)
    return img

class DetectorType(IntEnum):
    CENTER_FACE = 0
    S3FD = 1
    YOLOV5 = 2
DetectorTypeNames = ['CenterFace', 'S3FD', 'YoloV5']

class FaceSortBy(IntEnum):
    LARGEST = 0
    DIST_FROM_CENTER = 1
FaceSortByNames = ['@FaceDetector.largest', '@FaceDetector.dist_from_center']

class MarkerType(IntEnum):
    OPENCV_LBF = 0
    GOOGLE_FACEMESH = 1
MarkerTypeNames = ['OpenCV LBF','Google FaceMesh']


class FaceMaskType(IntEnum):
    SRC = 0
    CELEB = 1
    SRC_M_CELEB = 2
FaceMaskTypeNames = ['@FaceMerger.FaceMaskType.SRC','@FaceMerger.FaceMaskType.CELEB','@FaceMerger.FaceMaskType.SRC_M_CELEB']



class MyFaceDetector():
    '''
    Face Detector
    '''
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.detector_type = cfg['detector_type']
        if self.detector_type == DetectorType.CENTER_FACE:
            self.CenterFace = onnx_models.CenterFace(device)
        elif self.detector_type == DetectorType.S3FD:
            self.S3FD = onnx_models.S3FD(device)
        elif self.detector_type == DetectorType.YOLOV5:
            self.YoloV5Face = onnx_models.YoloV5Face(device)

    def run(self, frame_name, frame_image):
        detector_type = self.detector_type
        if (detector_type == DetectorType.CENTER_FACE and self.CenterFace is not None) or \
                (detector_type == DetectorType.S3FD and self.S3FD is not None) or \
                (detector_type == DetectorType.YOLOV5 and self.YoloV5Face is not None):

            _, H, W, _ = ImageProcessor(frame_image).get_dims()

            rects = []
            if detector_type == DetectorType.CENTER_FACE:
                rects = self.CenterFace.extract(frame_image, threshold=self.cfg['threshold'],
                                                fixed_window=self.cfg['fixed_window_size'])[0]
            elif detector_type == DetectorType.S3FD:
                rects = self.S3FD.extract(frame_image, threshold=self.cfg['threshold'],
                                          fixed_window=self.cfg['fixed_window_size'])[0]
            elif detector_type == DetectorType.YOLOV5:
                rects = self.YoloV5Face.extract(frame_image, threshold=self.cfg['threshold'],
                                                fixed_window=self.cfg['fixed_window_size'])[0]

            # to list of FaceURect
            rects = [FaceURect.from_ltrb((l / W, t / H, r / W, b / H)) for l, t, r, b in rects]

            # sort
            if eval(self.cfg['sort_by']) == FaceSortBy.LARGEST:
                rects = FaceURect.sort_by_area_size(rects)
            elif eval(self.cfg['sort_by']) == FaceSortBy.DIST_FROM_CENTER:
                rects = FaceURect.sort_by_dist_from_center(rects)

            if len(rects) != 0:
                max_faces = self.cfg['max_faces']
                if max_faces != 0 and len(rects) > max_faces:
                    rects = rects[:max_faces]

                if self.cfg['temporal_smoothing'] != 1:
                    if len(self.temporal_rects) != len(rects):
                        self.temporal_rects = [[] for _ in range(len(rects))]

                _face_mark_list = []
                for face_id, face_rect in enumerate(rects):
                    if self.cfg['temporal_smoothing'] != 1:
                        if len(self.temporal_rects[face_id]) == 0:
                            self.temporal_rects[face_id].append(face_rect.as_4pts())

                        self.temporal_rects[face_id] = self.temporal_rects[face_id][
                                                       -self.cfg['temporal_smoothing']:]

                        face_rect = FaceURect.from_4pts(np.mean(self.temporal_rects[face_id], 0))

                    if face_rect.get_area() != 0:
                        face_mark = FaceMark()
                        face_mark.set_image_name(frame_name)
                        face_mark.set_face_urect(face_rect)
                        _face_mark_list.append(face_mark)

                return _face_mark_list


class MyFaceMarker():
    '''
    Face Marker
    '''
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.marker_type = cfg['marker_type']

        if self.marker_type == MarkerType.OPENCV_LBF:
            self.opencv_lbf = cv_models.FaceMarkerLBF()
        elif self.marker_type == MarkerType.GOOGLE_FACEMESH:
            self.google_facemesh = onnx_models.FaceMesh(device)
        self.temporal_lmrks = []

    def run(self, frame_image, face_mark_list):
        marker_type = self.marker_type

        is_opencv_lbf = marker_type == MarkerType.OPENCV_LBF and self.opencv_lbf is not None
        is_google_facemesh = marker_type == MarkerType.GOOGLE_FACEMESH and self.google_facemesh is not None

        if marker_type is not None:
            if all_is_not_None(frame_image) and (is_opencv_lbf or is_google_facemesh):
                if self.cfg['temporal_smoothing'] != 1 and len(self.temporal_lmrks) != len(face_mark_list):
                    self.temporal_lmrks = [ [] for _ in range(len(face_mark_list)) ]

                for face_id, face_mark in enumerate(face_mark_list):
                    face_mark_rect = face_mark.get_face_urect()
                    if face_mark_rect is not None:
                        # Cut the face to feed to the face marker
                        face_image, face_uni_mat = face_mark_rect.cut(frame_image,
                            self.cfg['marker_coverage'], 256 if is_opencv_lbf else 192 if is_google_facemesh else 0 )
                        _,H,W,_ = ImageProcessor(face_image).get_dims()

                        if is_opencv_lbf:
                            lmrks = self.opencv_lbf.extract(face_image)[0]
                        elif is_google_facemesh:
                            lmrks = self.google_facemesh.extract(face_image)[0]

                        if self.cfg['temporal_smoothing'] != 1:
                            if len(self.temporal_lmrks[face_id]) == 0:
                                self.temporal_lmrks[face_id].append(lmrks)
                            self.temporal_lmrks[face_id] = self.temporal_lmrks[face_id][-self.cfg['temporal_smoothing']:]

                            lmrks = np.mean(self.temporal_lmrks[face_id],0 )

                        if is_google_facemesh:
                            face_mark.set_face_pose(FacePose.from_3D_468_landmarks(lmrks))

                        if is_opencv_lbf:
                            lmrks /= (W,H)
                        elif is_google_facemesh:
                            lmrks = lmrks[...,0:2] / (W,H)

                        face_ulmrks = FaceULandmarks.create (FaceULandmarks.Type.LANDMARKS_68 if is_opencv_lbf else \
                                                             FaceULandmarks.Type.LANDMARKS_468 if is_google_facemesh else None, lmrks)

                        face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
                        face_mark.add_face_ulandmarks (face_ulmrks)


class MyFrameAdjuster():
    '''
    Frame Adjuster
    '''
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, frame_image):
        if frame_image is not None:
            frame_image_ip = ImageProcessor(frame_image)
            frame_image_ip.median_blur(5, self.cfg['median_blur_per'] / 100.0)
            frame_image_ip.degrade_resize(self.cfg['degrade_bicubic_per'] / 100.0,
                                          interpolation=ImageProcessor.Interpolation.CUBIC)

            frame_image = frame_image_ip.get_image('HWC')
            return frame_image


class MyFaceAligner():
    '''
    Face Aligner
    '''
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, frame_name, frame_image, face_mark_list):
        face_align_images = {}
        for face_id, face_mark in enumerate(face_mark_list):
            face_ulmrks = face_mark.get_face_ulandmarks_by_type(FaceULandmarks.Type.LANDMARKS_468)
            if face_ulmrks is None:
                face_ulmrks = face_mark.get_face_ulandmarks_by_type(FaceULandmarks.Type.LANDMARKS_68)

            head_yaw = None
            if self.cfg['head_mode']:
                face_pose = face_mark.get_face_pose()
                if face_pose is not None:
                    head_yaw = face_pose.as_radians()[1]

            if face_ulmrks is not None:
                face_image, uni_mat = face_ulmrks.cut(frame_image, self.cfg['face_coverage'], self.cfg['resolution'],
                                                      exclude_moving_parts=self.cfg['exclude_moving_parts'],
                                                      head_yaw=head_yaw,
                                                      x_offset=self.cfg['x_offset'],
                                                      y_offset=self.cfg['y_offset'])

                face_align_image_name = f'{frame_name}_{face_id}_aligned'

                face_align = FaceAlign()
                face_align.set_image_name(face_align_image_name)
                face_align.set_coverage(self.cfg['face_coverage'])
                face_align.set_source_face_ulandmarks_type(face_ulmrks.get_type())
                face_align.set_source_to_aligned_uni_mat(uni_mat)

                for face_ulmrks in face_mark.get_face_ulandmarks_list():
                    face_align.add_face_ulandmarks(face_ulmrks.transform(uni_mat))
                face_mark.set_face_align(face_align)
                face_align_images[face_align_image_name] = face_image
        return face_align_images


class MyFaceSwapper():
    '''
    Face Swapper
    '''
    def __init__(self, cfg, device):
        self.cfg = cfg
        # dfm_models = DFLive.get_available_models_info(cfg['dfm_models_path'])
        dfm_path_stem = os.path.splitext(os.path.basename(cfg['dfs_model_path']))[0]
        model = DFLive.DFMModel.DFMModelInfo(dfm_path_stem, model_path=Path(cfg['dfs_model_path']), )
        self.dfm_model_initializer = DFLive.DFMModel_from_info(model, device)

    def run(self, face_align_images, face_mark_list):
        if self.dfm_model_initializer is not None:
            events = self.dfm_model_initializer.process_events()
            events = self.dfm_model_initializer.process_events()

            self.dfm_model = events.dfm_model

            model_width, model_height = self.dfm_model.get_input_res()
            dfm_model = self.dfm_model

            face_align_masks, face_swaps, face_swap_masks = {}, {}, {}
            for i, face_mark in enumerate(face_mark_list):
                face_align = face_mark.get_face_align()
                if face_align is not None:
                    face_align_image_name = face_align.get_image_name()
                    face_align_image = face_align_images[face_align_image_name]
                    if face_align_image is not None:

                        pre_gamma_red = self.cfg['pre_gamma_red']
                        pre_gamma_green = self.cfg['pre_gamma_green']
                        pre_gamma_blue = self.cfg['pre_gamma_blue']

                        fai_ip = ImageProcessor(face_align_image)
                        if self.cfg['presharpen_amount'] != 0:
                            fai_ip.sharpen(factor=self.cfg['presharpen_amount'])

                        if pre_gamma_red != 1.0 or pre_gamma_green != 1.0 or pre_gamma_blue != 1.0:
                            fai_ip.adjust_gamma(pre_gamma_red, pre_gamma_green, pre_gamma_blue)
                        face_align_image = fai_ip.get_image('HWC')

                        celeb_face, celeb_face_mask_img, face_align_mask_img = dfm_model.convert(
                            face_align_image, morph_factor=self.cfg['morph_factor'])
                        celeb_face, celeb_face_mask_img, face_align_mask_img = celeb_face[0], \
                                                                               celeb_face_mask_img[0], \
                                                                               face_align_mask_img[0]

                        if self.cfg['two_pass']:
                            celeb_face, celeb_face_mask_img, _ = dfm_model.convert(celeb_face,
                                                                                   morph_factor=self.cfg['morph_factor'])
                            celeb_face, celeb_face_mask_img = celeb_face[0], celeb_face_mask_img[0]

                        face_align_mask = FaceMask()
                        face_align_mask.set_image_name(f'{face_align_image_name}_mask')
                        face_align.set_face_mask(face_align_mask)
                        face_align_masks[face_align_mask.get_image_name()] = face_align_mask_img

                        face_swap = FaceSwap()
                        face_swap.set_image_name(f"{face_align_image_name}_swapped")
                        face_align.set_face_swap(face_swap)
                        face_swaps[face_swap.get_image_name()] = celeb_face

                        face_swap_mask = FaceMask()
                        face_swap_mask.set_image_name(f'{face_swap.get_image_name()}_mask')
                        face_swap.set_face_mask(face_swap_mask)
                        face_swap_masks[face_swap_mask.get_image_name()] = celeb_face_mask_img

                return face_align_masks, face_swaps, face_swap_masks


class MyFaceMerger():
    '''
    Face Merger
    '''
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.is_gpu = False

        if device != 'CPU':
            self.is_gpu = True

            global cp
            import cupy as cp  # BUG eats 1.8Gb paging file per process, so import on demand
            cp.cuda.Device(device.get_index()).use()

            self.cp_mask_clip_kernel = cp.ElementwiseKernel('T x', 'T z', 'z = x < 0.004 ? 0 : x > 1.0 ? 1.0 : x',
                                                            'mask_clip_kernel')
            self.cp_merge_kernel = cp.ElementwiseKernel('T bg, T face, T mask', 'T z', 'z = bg*(1.0-mask) + face*mask',
                                                        'merge_kernel')
            self.cp_merge_kernel_opacity = cp.ElementwiseKernel('T bg, T face, T mask, T opacity', 'T z',
                                                                'z = bg*(1.0-mask) + bg*mask*(1.0-opacity) + face*mask*opacity',
                                                                'merge_kernel_opacity')

    def run(self, frame_name, frame_image, face_mark_list, face_aligns, face_swaps, face_align_masks, face_swap_masks):
        if frame_image is not None:
            frame_finals = {}
            for face_mark in face_mark_list:
                face_align = face_mark.get_face_align()
                if face_align is not None:
                    face_swap = face_align.get_face_swap()
                    face_align_mask = face_align.get_face_mask()

                    if face_swap is not None:
                        face_swap_mask = face_swap.get_face_mask()
                        if face_swap_mask is not None:

                            face_align_img = face_aligns[face_align.get_image_name()]
                            face_swap_img = face_swaps[face_swap.get_image_name()]

                            face_align_mask_img = face_align_masks[face_align_mask.get_image_name()]
                            face_swap_mask_img = face_swap_masks[face_swap_mask.get_image_name()]
                            source_to_aligned_uni_mat = face_align.get_source_to_aligned_uni_mat()

                            face_mask_type = self.cfg['face_mask_type']

                            if all_is_not_None(face_align_img, face_align_mask_img, face_swap_img, face_swap_mask_img, face_mask_type):
                                face_height, face_width = face_align_img.shape[:2]

                                if self.is_gpu:
                                    frame_image = cp.asarray(frame_image)
                                    face_align_mask_img = cp.asarray(face_align_mask_img)
                                    face_swap_mask_img = cp.asarray(face_swap_mask_img)
                                    face_swap_img = cp.asarray(face_swap_img)

                                frame_image_ip = ImageProcessor(frame_image).to_ufloat32()
                                frame_image, (_, frame_height, frame_width, _) = frame_image_ip.get_image('HWC'), frame_image_ip.get_dims()
                                face_align_mask_img = ImageProcessor(face_align_mask_img).to_ufloat32().get_image('HW')
                                face_swap_mask_img = ImageProcessor(face_swap_mask_img).to_ufloat32().get_image('HW')

                                aligned_to_source_uni_mat = source_to_aligned_uni_mat.invert()
                                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-self.cfg['face_x_offset'], -self.cfg['face_y_offset'])
                                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(self.cfg['face_scale'],self.cfg['face_scale'])
                                aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)

                                if face_mask_type == FaceMaskType.SRC:
                                    face_mask = face_align_mask_img
                                elif face_mask_type == FaceMaskType.CELEB:
                                    face_mask = face_swap_mask_img
                                elif face_mask_type == FaceMaskType.SRC_M_CELEB:
                                    face_mask = face_align_mask_img*face_swap_mask_img

                                # Combine face mask
                                face_mask_ip = ImageProcessor(face_mask).erode_blur(self.cfg['face_mask_erode'], self.cfg['face_mask_blur'], fade_to_border=True) \
                                                                        .warpAffine(aligned_to_source_uni_mat, frame_width, frame_height)
                                if self.is_gpu:
                                    face_mask_ip.apply( lambda img: self.cp_mask_clip_kernel(img) )
                                else:
                                    face_mask_ip.clip2( (1.0/255.0), 0.0, 1.0, 1.0)
                                frame_face_mask = face_mask_ip.get_image('HWC')

                                frame_face_swap_img = ImageProcessor(face_swap_img) \
                                                      .to_ufloat32().warpAffine(aligned_to_source_uni_mat, frame_width, frame_height).get_image('HWC')

                                # Combine final frame
                                opacity = self.cfg['face_opacity']
                                if self.is_gpu:
                                    if opacity == 1.0:
                                        frame_final = self.cp_merge_kernel(frame_image, frame_face_swap_img, frame_face_mask)
                                    else:
                                        frame_final = self.cp_merge_kernel_opacity(frame_image, frame_face_swap_img, frame_face_mask, opacity)
                                    frame_final = cp.asnumpy(frame_final)
                                else:
                                    if opacity == 1.0:
                                        frame_final = ne.evaluate('frame_image*(1.0-frame_face_mask) + frame_face_swap_img*frame_face_mask')
                                    else:
                                        frame_final = ne.evaluate('frame_image*(1.0-frame_face_mask) + frame_image*frame_face_mask*(1.0-opacity) + frame_face_swap_img*frame_face_mask*opacity')

                                # keep image in float32 in order not to extra load FaceMerger

                                merged_frame_name = f'{frame_name}_merged'
                                frame_finals[merged_frame_name] = frame_final
            return frame_finals



def inference(img, device: str, onnx_device: ORTDeviceInfo):
    '''
    Inference Pipeline
    '''
    ## run face detector
    face_detector = MyFaceDetector(cfg=config['FaceDetector'], device=onnx_device)
    _face_mark_list = face_detector.run(frame_name=filename, frame_image=img)
    print(len(_face_mark_list), _face_mark_list)

    ## run face marker
    face_marker = MyFaceMarker(cfg=config['FaceMarker'], device=onnx_device)
    face_marker.run(frame_image=img, face_mark_list=_face_mark_list)

    ## run face aligner
    face_aligner = MyFaceAligner(cfg=config['FaceAligner'])
    face_aligns = face_aligner.run(frame_name=filename, frame_image=img, face_mark_list=_face_mark_list)

    ## run face swapper
    face_swapper = MyFaceSwapper(cfg=config['FaceSwapper'], device=onnx_device)
    face_align_masks, face_swaps, face_swap_masks = face_swapper.run(face_align_images=face_aligns,
                                                                     face_mark_list=_face_mark_list)

    ## run frame adjuster
    frame_adjuster = MyFrameAdjuster(cfg=config['FrameAdjuster'])
    img = frame_adjuster.run(frame_image=img)

    ## run face merger
    face_merger = MyFaceMerger(cfg=config['FaceMerger'], device=device)
    frame_finals = face_merger.run(filename, img, _face_mark_list, face_aligns, face_swaps, face_align_masks,
                                   face_swap_masks)

    return list(frame_finals.values())[0]


if __name__ == '__main__':
    ## get onnx devices info
    # devices = get_available_devices_info()
    # print('devices: ', devices)
    onnx_device = get_cpu_device()
    device = "CPU"

    ## read default config
    config = yaml.load(open('config.yml', 'rb'), Loader=yaml.Loader)

    ## read image
    filepath = 'C:/Users/zhenyuanshen/Downloads/DeepFaceLive_DirectX12/userdata/samples/photos/000007.jpg'
    filename = os.path.splitext(os.path.basename(filepath))[0]
    img = read_image(filepath)

    ## inference
    res_img = inference(img, device, onnx_device)
    print('result: ', res_img.shape)

    ## visualize
    cv2.imshow('img_color', res_img)
    cv2.waitKey(0)
    cv2.imwrite(f'result_{filename}.jpg', np.asarray(res_img * 255, dtype=np.uint8))
    cv2.destroyAllWindows()