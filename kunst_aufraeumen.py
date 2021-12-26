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
from skimage.transform import rescale
import matplotlib.pyplot as plt
import cv2 as cv

# n_segments = 44
n_colors = 7
stack_gap = 10
bg_out = 240

plt.close('all')
img = io.imread(Path('in.jpeg').expanduser())
# segments = slic(img, n_segments=n_segments, sigma=3, convert2lab=True)
segments = felzenszwalb(img, scale=100, min_size=1500, sigma=1)
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


def get_median_colors(img, segments):
    ids = sorted(np.unique(segments))
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
    for i in range(len(np.unique(segments))):
        # s = np.where(
        # np.repeat(segments[:, :, np.newaxis], 3, axis=2) == 20, img, 0)
        contours = cv.findContours((segments == i).astype(np.uint8), 1, 1)
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

out = np.full((1000, 1600, 3), bg_out, dtype=np.uint8)

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
plt.imshow(mark_boundaries(img, segments))
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(out)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()
