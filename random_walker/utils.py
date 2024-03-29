import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import numpy as np
import os
import cv2

def evaluationRW(labels, img):
    if labels.shape != img.shape:
        print("Error due to shape difference")
        return
    c = 0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j]!=img[i,j]:
                c+=1
    return c/(labels.shape[0]*labels.shape[1])*100

def show_results_random_malk(image, markers, labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Noisy data')
    ax2.imshow(markers, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Markers')
    ax3.imshow(labels )
    ax3.axis('off')
    ax3.set_title('Segmentation')

    fig.tight_layout()
    plt.show()
    
def show_results_random_walk02(image, markers, labels, masque, texte):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(texte[0])
    
    ax2.imshow(markers, cmap='gray')
    ax2.axis('off')
    ax2.set_title(texte[1])
    
    ax3.imshow(masque)
    ax3.axis('off')
    ax3.set_title(texte[2])
    
    ax4.imshow(labels)
    ax4.axis('off')
    ax4.set_title(texte[3])
    
    fig.tight_layout()
    plt.show()
    
def show_all_results(image, markers, masque, labels, segnet, texte):
    fig, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(1, 5, figsize=(10, 5),
                                        sharex=True, sharey=True)
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(texte[0])
    
    ax2.imshow(markers, cmap='gray')
    ax2.axis('off')
    ax2.set_title(texte[1])
    
    ax3.imshow(masque)
    ax3.axis('off')
    ax3.set_title(texte[2])
    
    ax4.imshow(labels)
    ax4.axis('off')
    ax4.set_title(texte[3])
    
    ax5.imshow(segnet)
    ax5.axis('off')
    ax5.set_title(texte[4])

    fig.tight_layout()
    plt.show()
        
    """
    Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    
    
    """
    
    
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "aucune":
        return image
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy