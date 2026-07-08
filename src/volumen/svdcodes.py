import numpy as np


def get_pixels(frame):
    rows, cols, channels = frame.shape
    red = frame[:,:,0]
    green = frame[:,:,1]
    blue = frame[:,:,2]
    pixels = np.vstack([red.reshape(rows*cols), green.reshape(rows*cols), blue.reshape(rows*cols)])
    return pixels


def get_transformed_data(pixels):
    normalized_data = pixels
    mu = normalized_data.mean(axis=1)
    matrix = normalized_data - mu[:, np.newaxis]
    u_svd, s_svd, v_svd = np.linalg.svd(matrix, full_matrices=False)
    return v_svd[:2,:]


def get_codes(h, alpha, delta=0.001):
    h_max = h.max(axis=1)[:, np.newaxis]
    h_min = h.min(axis=1)[:, np.newaxis]
    m = (1 - 2 * delta) / (h_max - h_min)
    b = (1 - delta - m * h_max) / m
    f = m * h + b
    g = np.floor(f).astype(np.uint8)
    weights = np.array([alpha, 1]).astype(np.uint8)
    rows, cols = g.shape
    codes = (weights.dot(g).astype(np.uint8)).reshape(cols)
    return codes


def get_subset_color_index(subset_color):
    color_mu = np.floor(subset_color.mean(axis=1))
    distance = np.sum((subset_color - color_mu[:, np.newaxis])**2, axis=0)
    color_index = np.argmin(distance)
    return color_index


def get_segments(frame, pixels, codes):
    frame_rows, frame_cols, frame_channels = frame.shape
    segments = np.zeros_like(pixels)
    unique_codes = np.unique(codes)
    for this_code in unique_codes:
        subsets = pixels[:, codes==this_code]
        if subsets.shape[1]:
            color_index = get_subset_color_index(subsets)
            color = subsets[:, color_index]
            segments[:, codes==this_code] = color[:, np.newaxis]
    result = (segments.transpose()).reshape((frame_rows, frame_cols, frame_channels))
    # print('result', str(result.shape))
    return result



def get_segments_mosaic_view(frame, pixels, codes):
    frame_rows, frame_cols, frame_channels = frame.shape
    subset_threshold = (frame_rows * frame_cols) // 16
    segments_array = list()
    small_segments = np.zeros_like(pixels)
    unique_codes = np.unique(codes)
    for this_code in unique_codes:
        subset = pixels[:, codes==this_code]
        if subset.shape[1] > subset_threshold:
            segment = np.zeros_like(pixels)
            color_index = get_subset_color_index(subset)
            color = subset[:, color_index].reshape((frame_channels, 1))
            segment[:, codes==this_code] = color
            segment = (segment.transpose()).reshape((frame_rows, frame_cols, frame_channels))
            segments_array.append(segment)
            # print('  ',
            #       f'this_code {this_code:3d}',
            #       f'shape {subset.shape[1]:>10d} -',
            #       f'{color[0,0]:>3d} {color[1,0]:>3d} {color[2,0]:>3d}')
        elif subset.shape[1]:
            color_index = get_subset_color_index(subset)
            color = subset[:, color_index].reshape((frame_channels, 1))
            small_segments[:, codes==this_code] = color
            # print('+ ',
            #       f'this_code {this_code:3d}',
            #       f'shape {subset.shape[1]:>10d} -',
            #       f'{color[0,0]:>3d} {color[1,0]:>3d} {color[2,0]:>3d}')
        else:
            pass
    small_segments = (small_segments.transpose()).reshape((frame_rows, frame_cols, frame_channels))
    segments_array.append(small_segments)
    mosaic_view = np.vstack(segments_array)
    # print('mosaic_view', str(mosaic_view.shape))
    return mosaic_view

def svd_codes_segment_image(input_image, alpha_string, this_view_type):
    pixels = get_pixels(input_image)
    h = get_transformed_data(pixels)
    codes = get_codes(h, int(alpha_string))
    if this_view_type == "mosaic view":
        output_image = get_segments_mosaic_view(input_image, pixels, codes)
    else:
        output_image = get_segments(input_image, pixels, codes)
    return output_image
