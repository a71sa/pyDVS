cimport numpy as np


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def singlefilter(img,step,windowsize,tresh):
    img2= img
    for (x, y, window) in sliding_window(img2, stepSize=step, windowSize=windowsize):
        if(np.count_nonzero(window))>=tresh:
            pass
        else :
            img2[x,y]=0
    return img2