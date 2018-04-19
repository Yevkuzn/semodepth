% Copyright 2017. All rights reserved.
% Computer Vision Group, Visual Computing Institute
% RWTH Aachen University, Germany
% This file is part of the semodepth project.
% Authors: Yevhen Kuznietsov (yevkuzn@gmail.com OR Yevhen.Kuznietsov@esat.kuleuven.be)
% semodepth project is free software; you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation; either version 3 of the License, or any later version.
% semodepth project is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
% PARTICULAR PURPOSE. See the GNU General Public License for more details.


%Set paths to gt and to predicted depth maps
gt_path = '...';
pred_path = '...';

%Get list of all files in gt / prediction directories
gt_files = dir(fullfile(gt_path, '*.mat'));
pred_files = dir(fullfile(pred_path, '*.mat'));

%Initialize cumulative metric values
crms = 0;
n_pxls = 0; %Number of valid pixels
crmsl = 0;
cabs_rel = 0;
csq_rel = 0;

cacc1 = 0;
cacc2 = 0;
cacc3 = 0;

%Evaluate
for i = 1:1:numel(gt_files)
    %Load GT and corresponding predictions
    gt = load(fullfile(gt_path, gt_files(i).name));
    pred = load(fullfile(pred_path, gt_files(i).name));
    gt_depths = double(gt.depth);
    [h, w] = size(gt_depths);
    pred_depths = reshape(pred.mat(:), 96, 320);
    %Resize predictions to match GT resolution
    predictions = double(imresize(pred_depths, [h, w], 'bilinear'));
    %Cap predicted values (use either 80 or 50 m)
    predictions(predictions > 80.0) = 80.0;
    predictions(predictions < 1.0) = 1.0;
    %Shows number of processed images
    if mod(i, 100) == 0
        disp(i);
    end;
  
    mask = gt_depths > 0.01; %All pixels with valid GT
    crop = zeros(h, w);
    %Garg / Eigen crop for image resolution 370x1224
    crop(fix(h-218):fix(h-3), 44 :1180) = 1;
    mask = mask & logical(crop); %All pixels to be used for evaluation
    n_pxls = n_pxls + sum(mask(:)); 
    
    %Compute RMSE
    rms = (gt_depths(:) - predictions(:)).^2;
    rms(~mask) = 0;
    crms = crms + sum(rms);
    
    %Compute RMSE log
    rmsl = (log(gt_depths(:)) - log(predictions(:))).^2;
    rmsl(~mask) = 0;
    crmsl = crmsl + sum(rmsl);
    
    gt_depths(gt_depths < 0.5) = 0.5; %epsilon
    
    %Compute ARD
    abs_rel = abs(gt_depths(:) - predictions(:)) ./ gt_depths(:);
    abs_rel(~mask) = 0;
    cabs_rel = cabs_rel + sum(abs_rel);
    
    %Compute SRD
    sq_rel = ((gt_depths(:) - predictions(:)).^2) ./ gt_depths(:);
    sq_rel(~mask) = 0;
    csq_rel = csq_rel + sum(sq_rel);
    
    max_ratio = max(gt_depths ./ predictions, predictions ./ gt_depths);
    
    %Compute accuracies
    acc1 = double(max_ratio < 1.25 & mask);
    acc2 = double(max_ratio < 1.25^2 & mask);
    acc3 = double(max_ratio < 1.25^3 & mask);
    
    cacc1 = cacc1 + sum(acc1(:));
    cacc2 = cacc2 + sum(acc2(:));
    cacc3 = cacc3 + sum(acc3(:));
end


RMSE = sqrt(crms / n_pxls)
RMSE_log = sqrt(crmsl / n_pxls)
ABS_REL = cabs_rel / n_pxls
SQ_REL = csq_rel / n_pxls
ACCURACY1 = cacc1 / n_pxls
ACCURACY2 = cacc2 / n_pxls
ACCURACY3 = cacc3 / n_pxls
