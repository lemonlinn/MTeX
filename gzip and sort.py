

import gzip
    def process_gz(self, in_path, out_path):
        path_in = os.path.abspath(in_path)
        the_dirs = os.listdir(in_path)
        out_files = os.listdir(out_path)
        
        for in_name, out_name in zip(the_dirs, out_files):
            the_filenames = os.listdir(path_in + "/" + in_name)
            for word in the_filenames[:]:
                f = gzip.open(path_in + '/' + in_name + '/' + word) #default rb; if text mode -> rt
                f.close()


#https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
#hope this is helpful
import argparse
import imutils
import cv2
    def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
##	# handle if we need to sort in reverse
##	if method == "right-to-left" or method == "bottom-to-top":
##		reverse = True
##	# handle if we are sorting against the y-coordinate rather than
##	# the x-coordinate of the bounding box
##	if method == "top-to-bottom" or method == "bottom-to-top":
##		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
