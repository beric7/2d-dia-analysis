import numpy as np

# Custom Ostu Thresholding method
def custom_otsu(image):
    # Set total number of bins in the histogram
    bins_num = 256
    
    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if True:
        hist = np.divide(hist.ravel(), hist.max())

    # Create the histogram
    # plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)

    # Save the figure
    # plt.savefig('./histogram.png')

    # Check if the maximum value in the histogram is above the threshold
    # Normalize the histogram
    hist_normalized = hist / hist.sum()
    max_value = np.max(hist_normalized)
    # print(max_value)

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # print(inter_class_variance)
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    # print("Otsu's algorithm implementation thresholding result: ", threshold)