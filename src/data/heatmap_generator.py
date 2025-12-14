import numpy as np

def generate_gaussian_heatmap(img_height, img_width, x, y, sigma=5):
    heatmap = np.zeros((img_height, img_width), dtype=np.float32)
    if x < 0 or y < 0:  # missing landmark
        return heatmap
    
    xx, yy = np.meshgrid(np.arange(img_width), np.arange(img_height))  # Gaussian
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    heatmap = heatmap / heatmap.max() # normalize 
    return heatmap


def generate_landmark_heatmaps(img_height, img_width, coords, sigma=5):
    num_landmarks = len(coords)
    heatmaps = np.zeros((num_landmarks, img_height, img_width), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        heatmaps[i] = generate_gaussian_heatmap(img_height, img_width, x, y, sigma)

    return heatmaps
