import csv
import cv2
import numpy as np
import os
import tensorflow as tf

try:
    import openslide
except ImportError:
    openslide = None

from whole_slide_cnn.vahadane import VahadaneNormalizer

class Dataset(object):
    """ Data list CSV parser """
    def __init__(
        self,
        csv_path,
        slide_dir,
        slide_file_extension=".ndpi",
        target_size=None,
        resize_ratio=1.0,
        slide_reader="openslide",
        use_tcga_vahadane=False,
        snapshot_path=None,
    ):
        self.target_size = target_size
        self.resize_ratio = resize_ratio
        self.slide_reader = slide_reader
        self.use_tcga_vahadane = use_tcga_vahadane
        self.snapshot_path = snapshot_path

        # Parse CSV to get paths
        self.slide_path_list = []
        self.y_true_list = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                slide_name = row[0]
                y_true = int(row[1])
                slide_path = os.path.join(
                    slide_dir, 
                    "{}{}".format(slide_name, slide_file_extension)
                )
                assert os.path.exists(slide_path), \
                    "{} not found while parsing {}.".format(slide_path, csv_path)
                self.slide_path_list.append(slide_path)
                self.y_true_list.append(y_true)

    def __len__(self):
        return len(self.slide_path_list)

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        y_true = self.y_true_list[idx]

        if self.slide_reader == "openslide":
            img = _read_slide_openslide(slide_path, self.resize_ratio, self.target_size)
        else:
            raise NotImplementedError(
                "{} is not a supported slide reader.".format(self.slide_reader)
            )

        if self.use_tcga_vahadane:
            normalizer_tmuh = VahadaneNormalizer.load("misc/tmuh_vahadane.pkl")
            normalizer_tcga = VahadaneNormalizer.load("misc/tcga_vahadane.pkl")
            img = normalizer_tmuh.normalize(img, normalizer_tcga)

        if self.snapshot_path != None:
            os.makedirs(self.snapshot_path, exist_ok=True)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.snapshot_path, "dataset_snapshot.tiff"),
                img_bgr,
            )

        return img, y_true

    def get_slide_path(self, idx):
        return self.slide_path_list[idx]

    def get_y_true(self, idx):
        return self.y_true_list[idx]
    
def _read_slide_openslide(
    slide_path,
    resize_ratio,
    target_size=None,
):
    assert openslide != None, \
        "Please install beforehand or select another slide loader."

    slide = openslide.open_slide(slide_path)

    if "aperio.AppMag" in slide.properties:
        mag = int(slide.properties["aperio.AppMag"])
        if mag == 40:
            pix_dim = 0.25
        elif mag == 20:
            pix_dim = 0.5
        else:
            assert False
        hamamatsu_pix_dim = 0.46
        resize_ratio = resize_ratio * (pix_dim / hamamatsu_pix_dim)

    w, h = slide.dimensions
    if target_size == None:
        target_size = (int(w * resize_ratio), int(h * resize_ratio))

    def get_region(loc_w, loc_h, src_sz_w, src_sz_h, req_sz_w, req_sz_h):
        downsample_factor_w = src_sz_w / req_sz_w
        downsample_factor_h = src_sz_h / req_sz_h
        downsample_factor = min(downsample_factor_w, downsample_factor_h)

        target_level = 0
        for level in range(len(slide.level_downsamples)):
            if slide.level_downsamples[level] <= downsample_factor:
                target_level = level

        src_sz_w_downsampled = int(src_sz_w / slide.level_downsamples[target_level])
        src_sz_h_downsampled = int(src_sz_h / slide.level_downsamples[target_level])

        img_rgba = slide.read_region(
            location=(loc_w, loc_h),
            level=target_level,
            size=(src_sz_w_downsampled, src_sz_h_downsampled)
        )

        img_rgba = np.array(img_rgba)
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img_rgb, (req_sz_w, req_sz_h), interpolation=cv2.INTER_CUBIC)
        return img

    img = _read_slide(
        slide_path,
        get_region,
        (0, 0),
        (w, h),
        target_size,
        resize_ratio,
    )
    return img

def _read_slide(
    slide_path,
    get_region_fn,
    coord,
    src_sz,
    dst_sz,
    resize_ratio,
):
    loc_w, loc_h = coord
    src_sz_w, src_sz_h = src_sz
    dst_sz_w, dst_sz_h = dst_sz

    # Get resized image
    req_sz_w, req_sz_h = int(src_sz_w * resize_ratio), int(src_sz_h * resize_ratio)
    img = get_region_fn(loc_w, loc_h, src_sz_w, src_sz_h, req_sz_w, req_sz_h)
   
    # Do cropping and pirnt warning
    h, w, c = img.shape
    cropped = False
    if w > dst_sz_w:
        index = int(w - dst_sz_w) // 2
        img = img[:, index: (index + dst_sz_w), :]
        cropped = True
    if h > dst_sz_h:
        index = int(h - dst_sz_h) // 2
        img = img[index: (index + dst_sz_h), :, :]
        cropped = True

    if cropped:
        print('Slide {} with original size {}x{}, resized size {}x{} exceeds the target size {}x{}. Crop it.'.format(
            slide_path, src_sz_w, src_sz_h, w, h, dst_sz_w, dst_sz_h
        ))
     
    # Do pading
    pad_color = (255, 255, 255)
    h, w, c = img.shape
    if w < dst_sz_w:
        l = (int(dst_sz_w - w)) // 2
        r = dst_sz_w - w - l
    else:
        l = 0
        r = 0
    if h < dst_sz_h:
        t = (int(dst_sz_h - h)) // 2
        b = dst_sz_h - h - t
    else:
        t = 0
        b = 0
    img = cv2.copyMakeBorder(
        img, t, b, l, r,
        cv2.BORDER_CONSTANT, value=pad_color
    )

    return img
