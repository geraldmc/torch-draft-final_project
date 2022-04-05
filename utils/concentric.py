def crop(img, size):
    """
    Crop an image concentrically to desired size.
    :param img: Input image
    :param size: Required crop image size
    """
    (h, w, c) = img.shape
    x = int((w - size[0]) / 2)
    y = int((h - size[1]) / 2)
    return img[y:(y + size[1]), x:(x + size[0]), :]