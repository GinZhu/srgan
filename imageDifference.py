from skimage import measure, io, color, img_as_float
from os import listdir, makedirs
from os.path import join, isdir
import numpy as np


ori_results_dir = '/local/scratch/jz426/GuangYang/SuperResolution/srgan_ori_rgb_valid_samples/evaluate'
diff_imgs_dir = ori_results_dir.replace('evaluate', 'diffImages')
if not isdir(diff_imgs_dir):
    makedirs(diff_imgs_dir)
    print('made dir:', diff_imgs_dir)

records = open(ori_results_dir.replace('evaluate', 'records.txt'), 'w')
errors = []

all_pred_imgs = io.ImageCollection(join(ori_results_dir, '*valid_gen.png'))
all_gt_imgs = io.ImageCollection(join(ori_results_dir, '*valid_hr.png'))
all_bicubic_imgs = io.ImageCollection(join(ori_results_dir, '*valid_bicubic.png'))

# img_id = 0

for img_id in range(len(all_pred_imgs)):
    # mse, nrmse, psnr, ssim
    pred_img = all_pred_imgs[img_id]
    gt_img = all_gt_imgs[img_id]
    bi_img = all_bicubic_imgs[img_id]

    gt_img = img_as_float(gt_img)
    pred_img = img_as_float(pred_img)
    bi_img = img_as_float(bi_img)

    # error of pred_img
    mse = measure.compare_mse(gt_img, pred_img)
    nrmse = measure.compare_nrmse(gt_img, pred_img)
    psnr = measure.compare_psnr(gt_img, pred_img)
    ssim = measure.compare_ssim(gt_img, pred_img, multichannel=True)

    # error of bicubic img
    bi_mse = measure.compare_mse(gt_img, bi_img)
    bi_nrmse = measure.compare_nrmse(gt_img, bi_img)
    bi_psnr = measure.compare_psnr(gt_img, bi_img)
    bi_ssim = measure.compare_ssim(gt_img, bi_img, multichannel=True)

    print('mse:', mse, '\n', 'nrmse:', nrmse, '\n', 'psnr:', psnr, '\n', 'ssim:', ssim, '\n')
    records.write(str(img_id) + '\nmse: ' + str(mse)
                  + '\nnrmse: ' + str(nrmse)
                  + '\npsnr: ' + str(psnr)
                  + '\nssim: ' + str(ssim) + '\n\n')

    print('bi_mse:', bi_mse, '\n', 'bi_nrmse:', bi_nrmse, '\n', 'bi_psnr:', bi_psnr, '\n', 'bi_ssim:', bi_ssim, '\n')
    records.write(str(img_id) + '\nbi_mse: ' + str(bi_mse)
                  + '\nbi_nrmse: ' + str(bi_nrmse)
                  + '\nbi_psnr: ' + str(bi_psnr)
                  + '\nbi_ssim: ' + str(bi_ssim) + '\n\n')

    errors.append([mse, nrmse, psnr, ssim, bi_mse, bi_nrmse, bi_psnr, bi_ssim])
    # image differences
    diff_img = (gt_img - pred_img+1.)/2.
    bi_diff_img = (gt_img - bi_img+1.)/2.
    io.imsave(join(diff_imgs_dir, str(img_id)+'.png'), diff_img)
    io.imsave(join(diff_imgs_dir, str(img_id)+'_bi.png'), bi_diff_img)

errors = np.array(errors)
np.savez(ori_results_dir.replace('evaluate', 'errors.npz'),
         mse=errors[:, 0],
         nrmse=errors[:, 1],
         psnr=errors[:, 2],
         ssim=errors[:, 3],
         bi_mse=errors[:, 4],
         bi_nrmse=errors[:, 5],
         bi_psnr=errors[:, 6],
         bi_ssim=errors[:, 7])


