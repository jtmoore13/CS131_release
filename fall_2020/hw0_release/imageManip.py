import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64)/255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    out = None
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """       
    out = .5*(image**2)
    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))
    
    rows_ratio = output_rows/input_rows
    cols_ratio = output_cols/input_cols
    
    # 2. Populate the `output_image` array using values from `input_image`
    for r in range(output_rows):
        for c in range(output_cols):
            output_image[r, c, :] = input_image[int(r/rows_ratio), int(c/cols_ratio), :]
             
    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!
    new_x = np.cos(theta)*point[0] - np.sin(theta)*point[1]
    new_y = np.cos(theta)*point[1] + np.sin(theta)*point[0]
    
    return np.array([new_x, new_y])
    


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    center_x = input_image.shape[0]/2
    center_y = input_image.shape[1]/2

    for r in range(input_rows):
        for c in range(input_cols):
         
            rotated_x, rotated_y = rotate2d(np.array([r-center_x, c-center_y]), theta)
            if 0 <= rotated_x+center_x < input_image.shape[0] and 0 <= rotated_y+center_y < input_image.shape[1]:
                output_image[r, c, :] = input_image[int(rotated_x+center_x), int(rotated_y + center_y), :]
                
            
            
    # 3. Return the output image
    return output_image
