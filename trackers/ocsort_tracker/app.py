import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)

tf.disable_v2_behavior()

from pathlib import Path


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape, img_info, idx):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    # if patch_shape is not None:
    #     # correct aspect ratio to patch shape
    #     target_aspect = float(patch_shape[1]) / patch_shape[0]
    #     new_width = target_aspect * bbox[3]
    #     bbox[0] -= (new_width - bbox[2]) / 2
    #     bbox[2] = new_width

    # convert to top left, bottom right
    # bbox[2:] += bbox[:2]
    # bb = bbox.copy()

    # bbox[0] =

    bbox = bbox.astype(np.int)

    # clip at image boundaries
    # bbox[:2] = np.maximum(0, bbox[:2])
    # bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None

    if (
        bbox[0] < 0
        or bbox[1] < 0
        or bbox[2] > image.shape[1]
        or bbox[3] > image.shape[0]
    ):
        return None

    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]

    print("BOX ", bbox)
    print("Image ", image.shape)

    dirs = "./VIZ/{}".format(img_info[4][0].split(".")[0])
    # Path(dirs).mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(os.path.join(dirs, "{}.png".format(idx)), image)

    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):
    def __init__(
        self, checkpoint_filename, input_name="images", output_name="features"
    ):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name
        )
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name
        )

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x},
            out,
            batch_size,
        )
        return out


def create_box_encoder(
    model_filename, input_name="images", output_name="features", batch_size=32
):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes, img_info):
        image_patches = []
        for idx, box in enumerate(boxes):
            patch = extract_image_patch(image, box, image_shape[:2], img_info, idx)
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0.0, 255.0, image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        # return np.zeros((len(boxes), 128))
        print("PATCH SIZE  - ", image_patches.shape)
        return image_encoder(image_patches, batch_size)

    return encoder


encoder = create_box_encoder("./pretrained/mars-small128.pb")


def get_viz_feats(img_fp, bboxs, img_info):

    bgr_image = cv2.imread(img_fp, cv2.IMREAD_COLOR)
    # bgr_image = cv2.resize(bgr_image, (1440, 800))
    # bgr_image = np_img[0]
    # breakpoint()
    print("NUM BOXES ", len(bboxs))
    print(bboxs)
    features = encoder(bgr_image, bboxs, img_info)
    return features


# detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]
