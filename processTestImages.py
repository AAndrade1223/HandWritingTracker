import cv2
import datetime
import glob
import os

class processTestImages:

	def saveImage(self,img,iteration):
		filename = 'testImg' + str(iteration) + '.png'
  
		# Using cv2.imwrite() method 
		# Saving the image 
		cv2.imwrite(filename, img)  
	
	def processImage(self,image):
		# Convert the image to gray scale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blur = cv2.medianBlur(gray, 5)
		thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)


		# Specify structure shape and kernel size. 
		# Kernel size increases or decreases the area 
		# of the rectangle to be detected.
		# A smaller value like (10, 10) will detect 
		# each word instead of a sentence.
		rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

		# Applying dilation on the threshold image
		dilation = cv2.dilate(thresh, rect_kernel, iterations = 5)

		img_invert = cv2.bitwise_not(image)
		# Finding contours
		contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# Creating a copy of image
		im2 = img_invert.copy()
	
		# Looping through the identified contours
		# Then rectangular part is cropped and passed on
		# to pytesseract for extracting text from it
		# Extracted text is then written into the text file
		for it, cnt in enumerate(contours):
			x, y, w, h = cv2.boundingRect(cnt)
	
			# Drawing a rectangle on copied image
			rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
			# Cropping the text block for giving input to OCR
			cropped = im2[y:y + h, x:x + w]
			
			#resize image
			cropped=cv2.resize(cropped, (28, 28))
			self.saveImage(cropped,it)

	def processImageDirectory(self):
		startingDirectory= os.path.dirname(os.path.abspath(__file__))
		inDirectory = ".\\data\\working\\rawImages\\"
		inDirectory = os.path.abspath(inDirectory)
		todaynowstr = "{date:%Y-%m-%d_%H:%M:%S}\\".format( date=datetime.datetime.now() ).replace(':','_')
		outDirectory = os.path.join(".\\data\\working\\processedImages\\", todaynowstr)
		outDirectoryClass = os.path.join(outDirectory, "img\\")
		os.makedirs(outDirectoryClass)
		outDirectory = os.path.abspath(outDirectory)
		outDirectoryClass = os.path.abspath(outDirectoryClass)
	
		os.chdir(inDirectory)
		images = [cv2.imread(image) for image in glob.glob(inDirectory+"/*.jpg")]
		os.chdir(outDirectoryClass) 
		for image in images:
			self.processImage(image)
		os.chdir(startingDirectory)
		return outDirectory
		
#pti=processTestImages()
#strout=pti.processImageDirectory()