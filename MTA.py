import cv2
import numpy as np
import os
import argparse
import time
from scipy.spatial.kdtree import KDTree as KDTree

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'pbm', 'pgm', 'ppm', 'tiff', 'tif']
workers = 2


def regularize_gray(src_img, dst_accumu_hist):
    '''
    src_img - one channel image
    dst_hist - destination image's accumulated histogram ranging form 0.0 to 1.0
    '''
    h1, w1 = src_img.shape[0], src_img.shape[1]
    # calculate histogram
    src_hist = cv2.calcHist([src_img], [0], None, [256], [0, 255]).reshape(-1)
    src_hist = hist2accum(src_hist)
    tree = KDTree(dst_accumu_hist.reshape((-1, 1)))

    vals_new = src_hist[src_img]
    _, dst_vals = tree.query(vals_new.reshape((-1, 1)), k=1, workers=workers)

    dst_vals = dst_vals.astype(np.uint8).reshape((h1, w1))

    return dst_vals


def regularize_rgb(src_img, dst_accumu_hists):
    '''
    src_img - rgb image
    dst_hists - destination image's accumulated histogram ranging form 0.0 to 1.0
    '''
    h, w = src_img.shape[0], src_img.shape[1]
    img_regu = [regularize_gray(src_img[:, :, i], dst_accumu_hists[i]).reshape(
        (h, w, 1)) for i in range(3)]
    img_ret = np.concatenate((img_regu[0], img_regu[1], img_regu[2]), axis=2)
    return img_ret


def hist2accum(hist):
    total = np.sum(hist)
    hist2 = hist.copy()
    for i in range(1, 256):
        hist2[i] = hist2[i-1]+hist2[i]
    return hist2/total


def transform_img(img, mode=0):
    if mode == 0:  # resize
        return img

    elif mode == 1:  # center crop
        h, w = img.shape[0], img.shape[1]
        if h >= w:
            top = h//2-w//2
            bottom = h//2+w//2
            return img[top:bottom, :, :]
        else:
            left = w//2-h//2
            right = w//2+h//2
            return img[:, left:right, :]

    elif mode == 2:  # random square crop
        h, w = img.shape[0], img.shape[1]
        size = int(min(w, h) * 0.8)
        x_lt = np.random.randint(w-size)
        y_lt = np.random.randint(h-size)
        return img[y_lt:y_lt+size, x_lt:x_lt+size, :]

    else:
        raise NotImplementedError


def merge_them_all(img_dir, dst_img, grid_w, mode):
    t = time.time()

    dst_h, dst_w = dst_img.shape[0], dst_img.shape[1]

    grid_h = int(1.0*dst_h/dst_w*grid_w)
    # grid resolution
    grid_reso_w = int(1.0*dst_w/grid_w)
    grid_reso_h = int(1.0*dst_h/grid_h)

    dst_img = cv2.resize(dst_img, (grid_w*grid_reso_w, grid_h*grid_reso_h))
    total = grid_h * grid_w

    # load images and resize them
    imgs = os.listdir(img_dir)
    imgs = [os.path.join(img_dir, im)
            for im in imgs if im.split('.')[-1].lower() in IMG_FORMATS]
    num_img = len(imgs)
    if num_img == 0:
        print('Failed to load any image!')
        return
    print('Loaded {0} images in total!'.format(num_img))
    # imgs = np.array(
    #     [cv2.resize(transform_img(cv2.imread(img), mode=mode),
    #      (grid_reso_w, grid_reso_h)) for img in imgs])
    
    imgs = np.array([cv2.imread(img) for img in imgs])

    img_ret = np.zeros_like(dst_img)
    t2 = time.time()
    for r in range(grid_h):
        for c in range(grid_w):
            idx = r*grid_w+c
            # img_id = idx % num_img
            img_id = np.random.randint(num_img)
            dst_img_grid = dst_img[r*grid_reso_h:(r+1) *
                                   grid_reso_h, c*grid_reso_w:(c+1)*grid_reso_w, :]
            dst_hists = [cv2.calcHist([dst_img_grid], [i], None, [256], [
                                      0, 255]).reshape(-1) for i in range(3)]
            dst_hists_accumu = np.array([hist2accum(h) for h in dst_hists])
            img = cv2.resize(transform_img(imgs[img_id], mode=mode), (grid_reso_w, grid_reso_h))
            img = regularize_rgb(img, dst_hists_accumu)
            img_ret[r*grid_reso_h:(r+1)*grid_reso_h, c *
                    grid_reso_w:(c+1)*grid_reso_w, :] = img

            idx += 1
            percen = 1.0*idx/total
            sym_num = int(percen*100)
            dot_num = 100 - sym_num
            print('\rProgress: |', end='')
            print('#'*sym_num, end='')
            print('.'*dot_num, end='')
            print('|', end='')
            time_per_im = (time.time()-t2)/idx
            print(' [{0}/{1}] Estimated remaining time {2:.1f} s'.format(idx,
                                                                         total, time_per_im*(total-idx)), end='')

    print()
    print('Time cost is {0:.1f} s.'.format(time.time()-t))
    return img_ret


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True, type=str,
                        help='Directory containing all images')
    parser.add_argument('--dst_img', required=True, type=str,
                        help='Path to the target image')
    parser.add_argument(
        '-s', '--scale', help='Rescale the target image', default=1.0, type=float)
    parser.add_argument('-g', '--grids', type=int, default=120,
                        help='Specify grid num of the target image in the width direction')
    parser.add_argument('-o', '--result_img', type=str, default='result.jpg',
                        help='Specify the path to the generated image')
    parser.add_argument('-j', '--threads', type=int, default=2,
                        help='Thread number')
    parser.add_argument('-t', type=int, default=1,
                        help='Specify resize mode. 0: resize | 1: center square crop | 2: random square crop')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    assert args.result_img.split('.')[-1].lower() in IMG_FORMATS

    img_dir = args.img_dir
    dst_img = args.dst_img
    scale = args.scale
    grids = args.grids
    dst_img = cv2.imread(dst_img)
    dst_img = cv2.resize(dst_img, None, None, fx=scale, fy=scale)

    workers = args.threads

    img = merge_them_all(img_dir, dst_img, grids, args.t)
    cv2.imwrite(args.result_img, img)
