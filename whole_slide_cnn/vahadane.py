import itertools
import numpy as np
import skimage.color
import pickle

def normalize_dynamic_range(image, percentile=95):
    """
    Normalize the dynamic range of an RGB image to 0~255. If the dynamic ranges of patches 
    from a dataset differ, apply this function before feeding images to VahadaneNormalizer,
    e.g. hema slides.
    :param image: A RGB image in np.ndarray with the shape [..., 3].
    :param percentile: Percentile to get the max value.
    """
    max_rgb = []
    for i in range(3):
        value_max = np.percentile(image[..., i], percentile)
        max_rgb.append(value_max)
    max_rgb = np.array(max_rgb)

    new_image = (np.minimum(image.astype(np.float32) * (255.0 / max_rgb), 255.0)).astype(np.uint8)
    
    return new_image

def beer_lambert_transform(rgb):
    od = -np.log(np.maximum(rgb.astype(np.float32), 1.0) / 255.0)
    return od

def beer_lambert_inverse(od):
    rgb = np.exp(-od) * 255.0
    rgb = np.minimum(np.maximum(rgb, 0.0), 255.0)
    rgb = rgb.astype(np.uint8)
    return rgb

def get_spams():
    spams = __import__("spams")
    assert spams != None
    return spams

class VahadaneNormalizer(object):
    """
    Vahadane color normalizer.

    Given two lists of slides, S and T, follow the steps to normalize S (source) by T (target).
    1. Initiate two normalizer objects for T and S respectively. 
    2. Run fit() for the two normalizers. 
       If either slide list is too huge, do some random sampling to reduce computational time.
    3. Call normalize() of the normalizer T with S and its corresponding normalizer as the arguments.
       The return list is the normalized slide images of S.
    """

    def __init__(
        self, 
        n_stains=2, 
    ):
        """
        Constructor.
        :param n_stains: Number of stains.
        """

        self.n_stains = n_stains

        self.stain_matrix = None
        self.lut_matrix = None # (R * 256 * 256 + G * 256 + B) -> concentrations
        self.dynamic_range = None

    def fit(
        self, 
        images, 
        white_thres=0.8, 
        lambda_dl=0.1, 
        lambda_lasso=0.01, 
        dynamic_range_percentile=95
    ):
        """
        Get the stain_matrix.
        :param images: 
            A list of RGB images, 
            each of which is an np.ndarray with size [H, W, 3] and uint8 dtype. 
        :param white_thres: Theshold of lightness to rule out white color.
        :param lambda_dl: A parameter for dictionary learning.
        :param lambda_lasso: A parameter for RGB to concentrations.
        :param dynamic_range_percentile: Percentile for dynamic range calculation.
        """

        # Get spams library
        spams = get_spams()

        # Extract non-white color from images
        colors = []
        for image in images:
            image_lab = skimage.color.rgb2lab(image)
            mask = image_lab[:, :, 0] < white_thres * 100.0
            extracted = image[mask]
            colors.append(extracted)
        colors = np.concatenate(colors, axis=0)

        # Do Beer-Lambert transformation
        od = beer_lambert_transform(colors)

        # Do sparse NMF
        stain_matrix = spams.trainDL(
            od.T, 
            K=self.n_stains, 
            lambda1=lambda_dl, 
            mode=2,
            modeD=0,
            posAlpha=True, 
            posD=True, 
            verbose=False,
            iter=200,
            clean=True,
        )
        sort_idx = np.argsort(stain_matrix[0, :]) # Sort by red color descending
        stain_matrix = stain_matrix[:, sort_idx]
        self.stain_matrix = stain_matrix

        # Generate lut_matrix for fast inference
        rgb_ascending = np.arange(256 * 256 * 256, dtype=np.int32)
        rgb_ascending = np.stack(
            [
                rgb_ascending // (256 * 256),
                rgb_ascending // 256 % 256,
                rgb_ascending % 256,
            ],
            axis=-1
        )
        od = beer_lambert_transform(rgb_ascending)
        concentrations = spams.lasso(
            od.T, 
            D=self.stain_matrix, 
            mode=2, 
            lambda1=lambda_lasso, 
            pos=True,
        ).toarray().T

        self.lut_matrix = concentrations

        # Get dynamic range
        concentrations = self.image2concentrations(colors)
        dynamic_range = []
        for i in range(self.n_stains):
            value_max = np.percentile(concentrations[:, i], dynamic_range_percentile)
            dynamic_range.append(value_max)
        self.dynamic_range = np.array(dynamic_range)

    def save(self, filename):
        """
        Save the normalizer.
        :param filename: Filename.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load the normalizer.
        :param filename: Filename.
        """
        with open(filename, "rb") as f:
            normalizer = pickle.load(f)
        return normalizer

    def image2concentrations(self, image):
        """
        Convert a RGB image into stain concentrations.
        :param image: A RGB image as an np.ndarray with the shape [..., 3].
        """
        assert self.lut_matrix is not None

        image = image.astype(np.int32)
        image_rgb_flatten = (
            image[..., 0] * (256 * 256) +
            image[..., 1] * 256 +
            image[..., 2]
        )
        concentrations = np.take(
            self.lut_matrix,
            image_rgb_flatten,
            axis=0,
        )

        return concentrations

    def concentrations2image(self, concentrations):
        """
        Convert stain concentrations into a RGB image.
        :param concentrations: np.ndarray with the shape [..., self.num_stains].
        """
        assert self.stain_matrix is not None

        od = np.dot(concentrations, self.stain_matrix.T)
        image = beer_lambert_inverse(od)
        return image
        
    def _align_concentrations(self, others_concentrations, others_normalizer):
        normed = others_concentrations / others_normalizer.dynamic_range

        all_permutations = list(itertools.permutations(range(self.n_stains)))
        best_score = 0
        best_permutation = None
        for permutation in all_permutations:
            permed = np.take(others_normalizer.stain_matrix, permutation, axis=1)
            score = np.sum(permed * self.stain_matrix)
            if score > best_score:
                best_score = score
                best_permutation = permutation

        permed = np.take(normed, best_permutation, axis=-1)

        concentrations = permed * self.dynamic_range

        return concentrations

    def normalize(self, image, image_normalizer=None):
        """
        Normalize the color of an image.
        :param image: A RGB image as np.ndarray with the shape [..., 3].
        :param image_normalizer: (Optional) The normalizer associated with the image.
        """
        assert self.stain_matrix is not None

        if image_normalizer == None:
            image_normalizer = VahadaneNormalizer()
            image_normalizer.fit([image])

        concentrations = image_normalizer.image2concentrations(image)
        aligned = self._align_concentrations(concentrations, image_normalizer)
        normed_image = self.concentrations2image(aligned)

        return normed_image

