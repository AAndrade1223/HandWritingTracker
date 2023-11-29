import cv2
from datetime import datetime
import glob
import os

from torch.nn import qat
class processTestImages:

	def saveImage(self,img,iter):
		todaynowstr = self.getTodayNow()
		filename = 'testContour_' + todaynowstr + "_" + str(iter) + '.png'
		cv2.imwrite(filename, img)  
		print("image saved: " + filename)
	
	def processImage(self,image):
		# Convert the image to gray scale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blur = cv2.medianBlur(gray, 5)
		thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)

		# Specify structure shape and kernel size. 
		# Kernel size increases or decreases the area 
		# of the rectangle to be detected.
		rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

		# Applying dilation on the threshold image
		dilation = cv2.dilate(thresh, rect_kernel, iterations = 5)

		#Invert image
		img_invert = cv2.bitwise_not(image)
		
		# Finding contours
		contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# Creating a copy of image
		im2 = img_invert.copy()
	
		# Looping through the identified contours
		# Then rectangular part is cropped and passed on
		for iter,cnt in enumerate(contours):
			x, y, w, h = cv2.boundingRect(cnt)
	
			# Drawing a rectangle on copied image
			rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
			# Cropping the text block for giving input to OCR
			cropped = im2[y:y + h, x:x + w]
			
			#resize image
			cropped=cv2.resize(cropped, (28, 28))
			self.saveImage(cropped,iter)

	def processImageDirectory(self,inDirectory="",outDirectory=""):
		startingDirectory,inDirectory,outDirectory,outDirectoryClass = self.formatDirectories(inDirectory,outDirectory)		
		os.chdir(inDirectory)	
		#images = [cv2.imread(image) for image in glob.glob(inDirectory + "/*.*")]
		images = []
		imgFound=0
		for image in os.listdir(inDirectory):
			if image.endswith(".png") or image.endswith(".jpg"):
				print("image found: " + image)
				img = cv2.imread(image)
				images.append(img)
				imgFound+=1
			else:
				print("image format invalid: ")
		os.chdir(outDirectoryClass) 
		print("Images found: " + str(imgFound))
		imgProcessed = 0
		for image in images:
			self.processImage(image)
			imgProcessed+=1
		print("Test images processed: " + str(imgProcessed))
		os.chdir(startingDirectory)
		return outDirectory
	
	def formatDirectories(self,inDirectory="",outDirectory=""):
		#Starting (original) directory
		startingDirectory= os.path.dirname(os.path.abspath(__file__))
		#test image "in" directory
		if not inDirectory:
			inDirectory = ".\\data\\working\\rawImages\\"
			inDirectory = os.path.abspath(inDirectory)
		#results "out" directory
		todaynowstr = self.getTodayNow()
		if not outDirectory:
			outDirectory = os.path.join(".\\data\\working\\processedImages\\", todaynowstr)
		outDirectory = os.path.join(outDirectory, todaynowstr)
		outDirectoryClass = os.path.join(outDirectory, "img\\")
		os.makedirs(outDirectoryClass)
		outDirectory = os.path.abspath(outDirectory)
		outDirectoryClass = os.path.abspath(outDirectoryClass)
		return startingDirectory,inDirectory,outDirectory,outDirectoryClass
	
	def getTodayNow(self):
		return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f').replace(':','_')
		