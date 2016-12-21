
import os
import sys

import numpy as np

from PIL import Image, ImageDraw


def square_coords(img_size, radius=5):
    xs = np.arange(radius, img_size[1] - radius - 1)
    ys = np.arange(radius, img_size[0] - radius - 1)
    x = np.random.choice(xs, size=1)[0]
    y = np.random.choice(ys, size=1)[0]
    coords = (x - radius, y - radius, x - radius, y + radius,
              x + radius, y + radius, x + radius, y - radius)
    return coords


def circle_coords(img_size, radius=5):
    xs = np.arange(radius, img_size[1] - radius - 1)
    ys = np.arange(radius, img_size[0] - radius - 1)
    x = np.random.choice(xs, size=1)[0]
    y = np.random.choice(ys, size=1)[0]
    coords = (x - radius, y - radius, x + radius, y + radius)
    return coords


def triangle_coords(img_size, width=7):
    # Half width
    hw = width / 2
    xs = np.arange(hw, img_size[1] - hw - 1)
    ys = np.arange(hw, img_size[0] - hw - 1)
    x = np.random.choice(xs, size=1)[0]
    y = np.random.choice(ys, size=1)[0]
    coords = (x - hw, y + hw, x, y - hw,
              x + hw, y + hw)
    return coords


def generate(out_dir, num_images=5000, img_size=(32, 32), square_radius=5,
             circ_radius=5, triangle_width=7):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    master_keys = [0, 1, 2, 3]
    quad_vals = [0, 1, 2, 3]
    for ix in range(num_images):
        sys.stdout.write("\r%d of %d" % (ix + 1, num_images))
        sys.stdout.flush()
        raw_file = os.path.join(out_dir, 'raw_' +
                                str(ix).zfill(len(str(num_images))) + '.png')
        img = Image.new('L', img_size)
        draw = ImageDraw.Draw(img)
        nkeys = np.random.choice(master_keys, 1)[0]
        if nkeys == 0:
            img.save(raw_file)
            img.close()
            continue
        keys = np.random.choice(master_keys, nkeys)
        if len(keys) == 1:
            quad_size = img_size
            quads = [[0, 0], [0, 0], [0, 0], [0, 0]]
        else:
            quad_size = (img_size[0] / 2, img_size[1] / 2)
            quads = [[0, 0],
                     [quad_size[1], 0],
                     [quad_size[1], quad_size[0]],
                     [0, quad_size[0]]]
        quads = np.array(quads)
        start_pts = np.random.choice(quad_vals, nkeys, replace=False)
        for ik, key in enumerate(keys):
            # Draw a square
            if key == 1:
                start_pt = quads[start_pts[ik]]
                coords = np.array(square_coords(quad_size, square_radius))
                coords[::2] += start_pt[0]
                coords[1::2] += start_pt[1]
                coords = coords.tolist()
                draw.polygon(coords, fill=100)
            elif key == 2:
                start_pt = quads[start_pts[ik]]
                coords = np.array(circle_coords(quad_size, circ_radius))
                coords[::2] += start_pt[0]
                coords[1::2] += start_pt[1]
                coords = coords.tolist()
                draw.ellipse(coords, fill=175)
            elif key == 3:
                start_pt = quads[start_pts[ik]]
                coords = np.array(triangle_coords(quad_size, triangle_width))
                coords[::2] += start_pt[0]
                coords[1::2] += start_pt[1]
                coords = coords.tolist()
                draw.polygon(coords, fill=255)
        img.save(raw_file)
        img.close()
