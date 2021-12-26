from pathlib import Path

from tqdm import tqdm
import numpy as np
import skimage.io as io
from skimage.segmentation import (
    mark_boundaries,
    slic,
)
from skimage.future import graph
from skimage.util import img_as_ubyte
import napari
from util import weight_mean_color, merge_mean_color

img = io.imread(Path('in.jpeg').expanduser())


range0 = [400]
range1 = [25, 50, 75]
range2 = [10]

fragments = np.zeros(
    (len(range0),
     len(range1),
     len(range2),
     ) + img.shape,
    dtype=np.uint8)

segments = np.zeros(
    (len(range0),
     len(range1),
     len(range2),
     ) + img.shape,
    dtype=np.uint8)

for i, ip in enumerate(tqdm(range0)):
    for j, jp in enumerate(tqdm(range1, leave=False)):
        for k, kp in enumerate(tqdm(range2, leave=False)):
            # frag = slic(
            # img, compactness=kp, n_segments=400, sigma=jp, start_label=1,
            # convert2lab=True)
            frag = slic(img, slic_zero=True, start_label=1)
            g = graph.rag_mean_color(img, frag)
            seg = graph.merge_hierarchical(
                frag,
                g,
                thresh=jp,
                rag_copy=False,
                in_place_merge=True,
                merge_func=merge_mean_color,
                weight_func=weight_mean_color,
            )

            fragments[i, j, k] = img_as_ubyte(mark_boundaries(img, frag))
            segments[i, j, k] = img_as_ubyte(mark_boundaries(img, seg))

v = napari.Viewer()
v.add_image(np.squeeze(fragments), visible=False)
v.add_image(np.squeeze(segments))
