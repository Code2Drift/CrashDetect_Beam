import cv2 as cv
import numpy as np
from numba import jit, prange
import cProfile
import pstats
import io


def profile(func):
    """A decorator that uses cProfile to profile a function"""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper


@jit(nopython=True, parallel=True)
def hoof_GPT(magnitude, angle):
    sum = np.zeros(9)
    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e. for each image pair
        for mag, ang in zip(magnitude[idx].reshape(-1), angle[idx].reshape(-1)):
            if ang >= 360:
                ang = ang - 360  # Make sure angles are within [0, 360)
            bin_idx = int( ang // 45 )
            sum[bin_idx] += mag
    return sum[0:8]


@profile
def show_frames(frame_num, cap, resolution):
    if not cap.isOpened():
        print("Error in opening video file")

    next_frame = frame_num + 1
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    status, frame_num = cap.read()
    _, next_frame = cap.read()

    return frame_num, next_frame

@profile
def resize_frame(frame, resolution):
    select_res = {
        360 : (640, 360),
        480 : (854, 480),
        720 : (1280, 720)
    }
    tuple_resolution = select_res.get(resolution)
    frame_width, frame_height = tuple_resolution
    frame_resized = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
    return frame_resized

@profile
def INIT_DenseOF(firstFrame, secondFrame):
    GR_firstFrame = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
    GR_secondframe = cv.cvtColor(secondFrame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(GR_firstFrame, GR_secondframe, None,
                                       0.5, 5, 15, 10, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    return GR_firstFrame, GR_secondframe, flow

@profile
def AFTER_DenseOF(firstFrame, secondFrame, opt_flow):
    GR_firstFrame = firstFrame
    GR_secondframe = cv.cvtColor(secondFrame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(GR_firstFrame, GR_secondframe, opt_flow,
                                       0.5, 5, 15, 10, 5, 1.1, cv.OPTFLOW_USE_INITIAL_FLOW)
    return GR_firstFrame, GR_secondframe, flow

@profile
def VIZ_OF_DENSE(flow, frame_2):
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    ang = (360 - ang) % 360
    mag = np.round(mag, 2)
    ang = ang.astype(int)
    hsv_mask = np.zeros_like(frame_2)
    hsv_mask[:, :, 1] = 255
    hsv_mask[:, :, 0] = ang / 2
    hsv_mask[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr_img = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)
    return bgr_img, mag, ang

@profile
def cut_OF(ORG_OF_frame, point_1, point_2):
    mask = np.zeros_like(ORG_OF_frame)
    mask[point_1[1]:point_2[1], point_1[0]:point_2[0]] = 1
    OF_cut = ORG_OF_frame * mask
    return OF_cut

@profile
def video_selection(scenario, case):
    score = str(scenario)
    user_input = str(case)
    str_input = score + user_input

    video_paths = {
        ## all path for scenario 1
        "1c1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\50-38.mp4",
        "1c2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\44-32.mp4",
        "1c3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\40-30.mp4",
        "1n1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\NT-38-50.mp4",
        "1n2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\NT-34-46.mp4",
        "1n3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\NT-28-40",
        "1m1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash_8\Wagon-etk800\NM_40-28.mp4",

        ## all path for scenario 2
        "2c1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\52-26.mp4",
        "2c2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\48-24.mp4",
        "2c3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\42-21.mp4",
        "2n1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\NT-35-50.mp4",
        "2n2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\NT-31-50.mp4",
        "2n3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\NT-25-50.mp4",
        "2m1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC2\Wagon-etk800\NM_50-35.mp4",

        ## all path for scenario 3
        "3c1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\50-40.mp4",
        "3c2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\46-38.mp4",
        "3c3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\40-32.mp4",
        "3n1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\NT-30-55.mp4",
        "3n2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\NT-30-51.mp4",
        "3n3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\NT-30-45.mp4",
        "3m1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC3\Wagon-etk800\NM_36-36.mp4",

        ## all path for scenario 4
        "4c1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\47-32.mp4",
        "4c2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\43-32.mp4",
        "4c3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\37-32.mp4",
        "4n1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\NT-36-46.mp4",
        "4n2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\NT-32-42.mp4",
        "4n3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Wagon-etk800\NT-26-36.mp4",
        "4m1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC4\Roamer4WD-Truck\NM.mp4",


        ## all path for sceanrio 5
        "5c1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\38-38.mp4",
        "5c2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\38-32.mp4",
        "5c3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\38-28.mp4",
        "5n1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\NT-38-36.mp4",
        "5n2": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\NT-38-32.mp4",
        "5n3": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\NT-38-28.mp4",
        #"5m1": r"E:\Capture\BeamNG_dataset\BeamNG.drive\Crash8_SC5\LegranSE-etk800\46-34.mp4",

    }

    return video_paths.get(str_input, "Video not found"), str_input