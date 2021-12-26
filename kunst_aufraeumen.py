from pathlib import Path

import numpy as np
import skimage.io as io
from sklearn.cluster import KMeans
from skimage.segmentation import (
    slic, mark_boundaries, felzenszwalb, watershed
)
from skimage.filters import sobel
from skimage.util import img_as_ubyte
from skimage.color import rgb2hsv, rgb2gray, rgba2rgb, rgb2lab
from skimage.transform import rescale, resize
from skimage.future import graph
import matplotlib.pyplot as plt
import cv2 as cv

from util import merge_mean_color, weight_mean_color

path = Path('keith_haring_photo.png').expanduser()

n_segments = 600
n_colors = 10
stack_gap = 5
bg_out = 240
merge_thres = 50

min_size = 100
rescale_factor = 0.2


plt.close('all')
img = io.imread(path)
img = rescale(img, scale=rescale_factor, multichannel=True)
img = img_as_ubyte(rgba2rgb(img))

frag = slic(
    img,
    compactness=10,
    n_segments=n_segments,
    start_label=1,
    sigma=0,
    convert2lab=True,
    multichannel=True,
    mask=rgb2gray(img) > 0.2,
)
# frag = slic(img, slic_zero=True)
# frag = felzenszwalb(img, scale=100, min_size=min_size, sigma=1)
plt.figure()
plt.imshow(mark_boundaries(img, frag))
plt.axis('off')
plt.show()

g = graph.rag_mean_color(img, frag)

segments = graph.merge_hierarchical(
    frag,
    g,
    thresh=merge_thres,
    rag_copy=False,
    in_place_merge=True,
    merge_func=merge_mean_color,
    weight_func=weight_mean_color,
)


# gradient = sobel(rgb2gray(img))
# segments = watershed(
    # gradient,
    # markers=n_segments,
    # compactness=0,
    # watershed_line=True
# )


# img = io.imread(Path('~/Desktop/keith_haring_photo.png').expanduser())
# img = rescale(img, 0.25, anti_aliasing=False, multichannel=True)
# img = img_as_ubyte(rgba2rgb(img))
# gradient = sobel(rgb2gray(img))
# segments = watershed(
    # gradient,
    # markers=n_segments,
    # compactness=1,
    # watershed_line=True,
    # mask=rgb2gray(img) > 25,
# )

print(f"{len(np.unique(segments))=}")


def filter_min_size(segments, min_size):
    """Sets all segment ids below min_size to 0."""
    ids, counts = np.unique(segments, return_counts=True)
    for i, c in zip(ids, counts):
        if c < min_size:
            segments[segments == i] = 0
    return segments


segments = filter_min_size(segments, min_size)

plt.figure()
plt.imshow(mark_boundaries(img, segments))
plt.axis('off')
plt.show()


def get_median_colors(img, segments):
    ids = sorted(np.unique(segments))
    ids.remove(0)  # remove background
    median_colors = []
    for i in ids:
        element = img[segments == i]
        color = np.median(element, axis=0)
        median_colors.append(color)
    return np.array(median_colors)


median_colors = get_median_colors(img, segments)
median_colors_hsv = rgb2hsv(median_colors)

model = KMeans(n_colors, n_init=100)
clusters = model.fit_predict(median_colors)
clustered_ids = []
for i in range(n_colors):
    clustered_ids.append(np.where(clusters == i))

medians = []
for i in range(n_colors):
    idx = np.where(clusters == i)[0]
    meds = []
    for j in idx:
        meds.append(median_colors[j])
    meds = np.array(meds)
    c_median = np.median(meds, axis=0)
    medians.append(c_median)

medians = np.array(medians)

med_to_sort = np.array(
    list(zip(medians[:, 0], medians[:, 1], medians[:, 2])), dtype=[
        ('hue', np.uint8), ('saturation', np.uint8), ('value', np.uint8)])
indices = np.argsort(med_to_sort, order=['hue', 'saturation', 'value'])


def get_crops(img, segments):
    crops = []
    for i in np.unique(segments):
        if i == 0:
            continue
        # s = np.where(
        # np.repeat(segments[:, :, np.newaxis], 3, axis=2) == 20, img, 0)
        contours = cv.findContours(
            (segments == i).astype(np.uint8),
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE,
        )
        # opencv flips x and y
        y, x, w, h = cv.boundingRect(contours[0][0])
        imgc = img.copy()
        crop = imgc[x:x + h, y:y + w]

        mask_full = segments == i
        mask_crop = mask_full[x:x + h, y:y + w]
        crop[~mask_crop] = np.array([bg_out, bg_out, bg_out])
        crops.append(crop)
    return crops


crops = get_crops(img, segments)
print(f"{len(crops)=}")


clustered_crops = []
for i in range(n_colors):
    idx = np.where(clusters == i)[0]
    ccrops = []
    for j in idx:
        ccrops.append(crops[j])

    clustered_crops.append(ccrops)


sorted_crops = []
for i in indices:
    sorted_crops.append(clustered_crops[i])
clustered_crops = sorted_crops
clustered_crops.reverse()

total_crops = 0
for croplist in clustered_crops:
    for i in croplist:
        total_crops += 1
print(f"{total_crops=}")

out = np.full((1200, 1500, 3), bg_out, dtype=np.uint8)

stack_centers = np.linspace(
    0,
    out.shape[1],
    num=n_colors + 2,
    endpoint=True)[
        1: -1]

assert len(stack_centers) == len(clustered_crops), \
    f"Num stacks: {len(clustered_crops)}\nNum stack centers: {len(stack_centers)}"


for sc, crops in zip(stack_centers, clustered_crops):
    height = out.shape[0] - stack_gap
    for c in crops:
        hori_pos = int(sc) - c.shape[1] // 2
        if height - c.shape[0] <= 0:
            raise ValueError("Increase output image height.")
        out[height - c.shape[0]: height, hori_pos: hori_pos + c.shape[1], :] = c
        height = height - c.shape[0] - stack_gap


plt.figure()
plt.imshow(out)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()
