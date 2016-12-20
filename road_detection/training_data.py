
import json
import os
import sys

import osgeo.gdal as gdal
import numpy as np
import scipy.misc as misc


def generate_envi(road_file, image_file, meta_file, out_dir, img_buff=128,
                  num_images=5000, color_scales=[2000.0, 2800.0, 2000.0],
                  img_scale=0.5):
    road_mask = misc.imread(road_file)
    road_mask = (road_mask[:, :, -1] > 0).astype('uint8') * 255

    with open(meta_file, 'rb') as fid:
        meta_data = json.load(fid)

    x_range = meta_data['xrange']
    y_range = meta_data['yrange']
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]

    ds = gdal.Open(image_file)
    images = []
    for band, scale in zip([2, 3, 5], color_scales):
        rs = ds.GetRasterBand(band)
        pan_chip = rs.ReadAsArray(xoff=x_range[0] + 2, yoff=y_range[0] + 1,
                                  win_xsize=x_size, win_ysize=y_size)
        pan_chip = pan_chip.astype('float32') / scale
        pan_chip[pan_chip > 1.0] = 1.0
        images.append(misc.bytescale(pan_chip, 0, 1))

    yinds, xinds = np.where(road_mask > 0)

    good_inds = np.where((yinds > img_buff) & (yinds < (y_size - img_buff)) &
                         (xinds > img_buff) & (xinds < (x_size - img_buff)))[0]
    yinds = yinds[good_inds]
    xinds = xinds[good_inds]

    indices = np.arange(len(yinds))

    road_inds = np.random.choice(indices, size=num_images, replace=False)

    out_size = img_buff * 2 * img_scale

    for ix, road_ind in enumerate(road_inds):
        xs = xinds[road_ind] - img_buff
        xe = xinds[road_ind] + img_buff
        ys = yinds[road_ind] - img_buff
        ye = yinds[road_ind] + img_buff
        raw_chip = np.zeros((out_size, out_size, 3), dtype='uint8')
        raw_chip[:, :, 0] = misc.imresize(images[2][ys:ye, xs:xe].copy(),
                                          img_scale)
        raw_chip[:, :, 1] = misc.imresize(images[1][ys:ye, xs:xe].copy(),
                                          img_scale)
        raw_chip[:, :, 2] = misc.imresize(images[0][ys:ye, xs:xe].copy(),
                                          img_scale)
        truth_chip = misc.imresize(road_mask[ys:ye, xs:xe], img_scale)
        raw_file = os.path.join(out_dir, 'raw_' +
                                str(ix).zfill(len(str(num_images))) + '.png')
        truth_file = os.path.join(out_dir, 'truth_' +
                                  str(ix).zfill(len(str(num_images))) + '.png')
        # np.save(raw_file, raw_chip)
        # np.save(truth_file, truth_chip)
        misc.imsave(raw_file, raw_chip)
        misc.imsave(truth_file, truth_chip)


def generate_grayscale(road_file, image_file, out_dir, img_buff=128,
                       num_images=5000):
    road_mask = misc.imread(road_file)
    road_mask = (road_mask[:, :, -1] > 0).astype('uint8') * 255

    image = misc.imread(image_file)

    y_size, x_size = image.shape

    yinds, xinds = np.where(road_mask > 0)

    good_inds = np.where((yinds > img_buff) & (yinds < (y_size - img_buff)) &
                         (xinds > img_buff) & (xinds < (x_size - img_buff)))[0]
    yinds = yinds[good_inds]
    xinds = xinds[good_inds]

    indices = np.arange(len(yinds))

    road_inds = np.random.choice(indices, size=num_images, replace=False)

    for ix, road_ind in enumerate(road_inds):
        xs = xinds[road_ind] - img_buff
        xe = xinds[road_ind] + img_buff
        ys = yinds[road_ind] - img_buff
        ye = yinds[road_ind] + img_buff
        raw_chip = image[ys:ye, xs:xe]
        truth_chip = road_mask[ys:ye, xs:xe]
        raw_file = os.path.join(out_dir, 'raw_' +
                                str(ix).zfill(len(str(num_images))) + '.png')
        truth_file = os.path.join(out_dir, 'truth_' +
                                  str(ix).zfill(len(str(num_images))) + '.png')
        # np.save(raw_file, raw_chip)
        # np.save(truth_file, truth_chip)
        misc.imsave(raw_file, raw_chip)
        misc.imsave(truth_file, truth_chip)
        sys.stdout.write("\r%d of %d" % (ix, len(road_inds)))
        sys.stdout.flush()
