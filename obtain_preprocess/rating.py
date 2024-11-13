# import the necessary modules:
from psychopy import visual, event
from datetime import datetime
import glob 
import csv
import os

# define the window. Notice that here we will get a window in full screen and with the color black
win = visual.Window(fullscr=True, color='black')

# set the path to the image directory
path = "frontalimages_manuallyaligne_greyscale/"

# get all the image names from the directory
images = glob.glob(path + "*.jpg")

# Generate a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp}.csv"

with open(filename, "w") as f:
    writer = csv.writer(f,delimiter=',')
    # then we loop through the images and for each image we also draw a rating scale  
    try:
        for i in images:
            # stimulus object 
            stimulus = visual.ImageStim(win, image = i, pos = [0,0.3]) 

            # create the rating scale:
            ratingScale = visual.RatingScale(win, 
                scale = None,          #This makes sure there's no subdivision on the scale.
                low = 1,               #This is the minimum value I want the scale to have.
                high = 9,             #This is the maximum value of the scale.
                singleClick = True,    #This allows the user to submit a rating by one click.
                showAccept = False,    #This shows the user's chosen value in a window below the scale.
                markerStart = 5,       #This sets the rating scale to have its marker start on 5.
                labels = ['Neutral Emotion', 'Positive Emotion'], #This creates the labels.
                pos = [0, -0.6])       #This sets the scale's position.

            # draw and show the image and the rating scale
            while ratingScale.noResponse:
                stimulus.draw()
                      win.flip()

                # Check for key presses
                keys = event.getKeys()
                if 'escape' in keys:  # Exit if ESC is pressed
                    raise KeyboardInterrupt
                if 'q' in keys:  # You can also use other keys if you prefer
                    raise KeyboardInterrupt

            # record the response, i.e. the rating 
            rating = ratingScale.getRating()
            # print rating to the console
            writer.writerow([os.path.basename(i), rating])

    except KeyboardInterrupt:
        print("Exiting program.")
# close window
win.close()
