#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
FLAGS = tf.flags.FLAGS


def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = tf.meshgrid(tf.range(offset + start, stop), tf.range(offset + start, stop))
  assert x.shape[0] == size
  g = tf.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / tf.reduce_sum(g)


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.

  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).

  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.shape.ndims != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.shape.ndims)

  img1 = tf.cast(img1, tf.float64)
  img2 = tf.cast(img2, tf.float64)
  _, height, width, _ = img1.shape.as_list()

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = tf.reshape(_FSpecialGauss(size, sigma), (size, size, 1, 1))
    window = tf.cast(tf.concat([window] * img1.shape[-1], axis=-2), tf.float64)

    def my_convolve(x, y, mode='VALID'):
        return tf.nn.depthwise_conv2d(x, y, strides=[1] * 4, padding=mode)

    mu1 = my_convolve(img1, window, mode='VALID')
    mu2 = my_convolve(img2, window, mode='VALID')
    sigma11 = my_convolve(img1 * img1, window, mode='VALID')
    sigma22 = my_convolve(img2 * img2, window, mode='VALID')
    sigma12 = my_convolve(img1 * img2, window, mode='VALID')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = tf.reduce_mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = tf.reduce_mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.

  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.

  Returns:
    MS-SSIM score between `img1` and `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.shape.ndims != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.shape.ndims)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = tf.ones((2, 2, 1, 1), dtype=tf.float64) / 4.0
  downsample_filter = tf.concat([downsample_filter] * img1.shape[-1], axis=-2)

  im1, im2 = [tf.cast(x, tf.float64) for x in [img1, img2]]
  mssim = []
  mcs = []

  def my_convolve_pad(x, y, mode='SYMMETRIC'):
    if mode == 'REFLECT':
        mode = 'SYMMETRIC'
    pad_x = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], mode=mode)
    return tf.nn.depthwise_conv2d(pad_x, y, strides=[1] * 4, padding='VALID')

  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim.append(ssim)
    mcs.append(cs)
    filtered = [my_convolve_pad(im, downsample_filter, mode='REFLECT')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
  return (tf.reduce_prod(tf.stack(mcs[0:levels-1]) ** weights[0:levels-1]) *
          (tf.stack(mssim[levels-1]) ** weights[levels-1]))
