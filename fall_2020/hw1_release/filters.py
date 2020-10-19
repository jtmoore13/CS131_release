"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    kernel = np.flip(kernel)
    
    for wi in range(Wi):
        for hi in range(Hi): 
            out_pixel = 0   
            # sum up all the overlapping pixels between the kernel and image
            for x in range(Wk):
                for y in range(Hk):
                    if  0 <= wi+x-Wk//2 < Wi and 0 <= hi+y-Hk//2 < Hi:
                        out_pixel += kernel[y][x]*image[hi+y-Hk//2][wi+x-Wk//2]
 
            out[hi][wi] = out_pixel
                                   
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape

    out = np.zeros((H+2*pad_height, W+2*pad_width))
    start_x = (W+2*pad_width)//2 - W//2
    start_y = (H+2*pad_height)//2 - H//2
    
    for x in range(W):
        for y in range(H):
            out[start_y+y][start_x+x] = image[y][x]
    
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    kernel = np.flip(kernel)
    padded_image = zero_pad(image, Hk//2, Wk//2)
    
    half_ix = Wi//2
    half_iy = Hi//2
    half_kx = Wk//2
    half_ky = Hk//2
    start_y = padded_image.shape[0]//2 - half_iy
    start_x = padded_image.shape[1]//2 - half_ix
  
    for x in range(padded_image.shape[1]):
        for y in range(padded_image.shape[0]):
            if start_x <= x <= padded_image.shape[1]-start_x-1 and start_y <= y <= padded_image.shape[0]-start_y-1:
                mult = np.multiply(kernel,padded_image[y-half_ky:y-half_ky+Hk, x-half_kx:x-half_kx+Wk])
                out[y-start_y][x-start_x] = np.sum(mult)

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_fast(f, np.flip(g))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    g_mean = np.mean(g)
    g = np.subtract(g, g_mean)
    
    return conv_fast(f, np.flip(g))

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    g_mean = np.mean(g)
    g_std = np.std(g)
    new_g = np.divide(np.subtract(g, g_mean), g_std)
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    padded_image = zero_pad(f, Hg//2, Wg//2)
    
    half_fx = Wf//2
    half_fy = Hf//2
    half_gx = Wg//2
    half_gy = Hg//2
    
    start_y = padded_image.shape[0]//2 - half_fy
    start_x = padded_image.shape[1]//2 - half_fx
  
    for x in range(padded_image.shape[1]):
        for y in range(padded_image.shape[0]):
            if start_x <= x <= padded_image.shape[1]-start_x-1 and start_y <= y <= padded_image.shape[0]-start_y-1:
                patch = padded_image[y-half_gy:y-half_gy+Hg, x-half_gx:x-half_gx+Wg]
                patch_mean = np.mean(patch)
                patch_std = np.std(patch)
                new_patch = np.divide(np.subtract(patch, patch_mean), patch_std)
          
                mult = np.multiply(new_g, new_patch)
                    
                out[y-start_y][x-start_x] = np.sum(mult)
                
    return out
    
    
    
    
    
    
    
    








