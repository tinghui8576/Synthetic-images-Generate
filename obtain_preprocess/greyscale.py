from PIL import Image
import os
from os import listdir

# get the path/directory
folder_dir = "frontalimages_manuallyaligned_only_male/"
for images in os.listdir(folder_dir):

	# check if the image ends with png
	if (images.endswith(".jpg")):
		print(images)
		p = os.path.join(folder_dir, images)
		img = Image.open(p).convert('L')
		
		path = 'frontalimages_manuallyaligne_greyscale/'
		img.save(path+images)