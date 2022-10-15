#%%

import os
import glob
import imageio
import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt
from ngp_config import *


img_paths = sorted(list(pathlib.Path(VID_DIR).glob('*.mp4')))
for img_path in img_paths:
    print('Processing ' + str(img_path))
    cap = cv2.VideoCapture(str(img_path))
    cap.set(1, MIN_FRAME)
    frames = []
    for cur_frame in range(MAX_FRAME - MIN_FRAME):
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IM_W, IM_H))
        frames.append(frame)
        #plt.imshow(frame)
        #plt.show()
        g = 2

    std = np.std(frames, axis = 0)
    std = np.mean(std, axis = -1)
    if not os.path.exists(ATEN_FOLDER):
        os.mkdir(ATEN_FOLDER)
    fname = os.path.basename(img_path).split('.')[0]
    np.save(os.path.join(ATEN_FOLDER, fname), std.astype(np.float16))
    #cv2.imwrite(os.path.join(ATEN_FOLDER, fname + '.png'), std)
    g = 2
