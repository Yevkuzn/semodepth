# Copyright 2017. All rights reserved.
# Computer Vision Group, Visual Computing Institute
# RWTH Aachen University, Germany
# This file is part of the semodepth project.
# Authors: Yevhen Kuznietsov (yevkuzn@gmail.com OR Yevhen.Kuznietsov@esat.kuleuven.be)
# semodepth project is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# semodepth project is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.


import os
import locale
import scipy.io as sio
import numpy as np
import cv2

#For sorting files in alphabetical order
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#Paths for GT and predictions
gt_path = '...'
pred_path = '...'

#Get the file names for evaluation
gt_files = []
for f in os.listdir(gt_path):
    if '.mat' in f:
        gt_files.append(f)
pred_files = []
for f in os.listdir(pred_path):
    if '.mat' in f:
        pred_files.append(f)

#Sort the file names
gt_files.sort(cmp=locale.strcoll)
pred_files.sort(cmp=locale.strcoll)

n_pxls = 0
#Initialize cumulative metric values
crms = 0
crmsl = 0
cabs_rel = 0
csq_rel = 0
cacc1 = 0
cacc2 = 0
cacc3 = 0

for i in xrange(0, len(gt_files)):
    #Load GT and prediction depth maps
    gt = sio.loadmat(os.path.join(gt_path, gt_files[i]))
    pred = sio.loadmat(os.path.join(pred_path, pred_files[i]))
    gt_depths = gt["depth"]
    h, w = np.shape(gt_depths)
    pred_depths = np.squeeze(pred['mat'], axis=[0, 3])
    #Resize prediction to match GT size
    pred_depths = cv2.resize(pred_depths, (w, h), interpolation=cv2.INTER_LINEAR)
    #Cap predicted depth, either [1, 80] or [1, 50]
    pred_depths[pred_depths > 80.0] = 80.0
    pred_depths[pred_depths < 1.0] = 1.0

    #Show the number of evaluated frames
    if i % 100 ==0:
        print(i)

    #Create mask of valid pixels
    mask = gt_depths > 0.01
    crop = np.zeros_like(pred_depths)
    crop[h-219:h-3, 44:1180] = 1
    mask = np.logical_and(mask, (crop == 1))

    #Count number of valid pixels
    n_pxls += np.sum(mask)

    #Compute RMSE
    rms = np.square(gt_depths - pred_depths)
    rms[~mask] = 0
    crms += np.sum(rms)

    gt_depths[gt_depths < 0.5] = 0.5 #epsilon

    #Compute RMSE log
    rmsl = np.square(np.log(gt_depths) - np.log(pred_depths))
    rmsl[~mask] = 0
    crmsl += np.sum(rmsl)

    #Compute ARD
    abs_rel = np.abs(gt_depths - pred_depths) / gt_depths
    abs_rel[~mask] = 0
    cabs_rel += np.sum(abs_rel)

    #Compute SRD
    sq_rel = np.square(gt_depths - pred_depths) / gt_depths
    sq_rel[~mask] = 0
    csq_rel += np.sum(sq_rel)

    max_ratio = np.maximum(gt_depths / pred_depths, pred_depths / gt_depths)

    #Compute accuracies for different deltas
    acc1 = np.asarray(np.logical_and(max_ratio < 1.25, mask), dtype = np.float32)
    acc2 = np.asarray(np.logical_and(max_ratio < 1.25 ** 2, mask), dtype = np.float32)
    acc3 = np.asarray(np.logical_and(max_ratio < 1.25 ** 3, mask), dtype = np.float32)

    cacc1 += np.sum(acc1)
    cacc2 += np.sum(acc2)
    cacc3 += np.sum(acc3)
    

RMSE = np.sqrt(crms / n_pxls)
RMSE_log = np.sqrt(crmsl / n_pxls)
ABS_REL = cabs_rel / n_pxls
SQ_REL = csq_rel / n_pxls
ACCURACY1 = cacc1 / n_pxls
ACCURACY2 = cacc2 / n_pxls
ACCURACY3 = cacc3 / n_pxls

#Display metrics
print(RMSE)
print(RMSE_log)
print(ABS_REL)
print(SQ_REL)
print(ACCURACY1)
print(ACCURACY2)
print(ACCURACY3)
print(n_pxls)
