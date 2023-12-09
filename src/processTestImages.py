import csv
import shutil
import cv2
from datetime import datetime
import os

class processTestImages:
	
	def getTodayNow(self):
		return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f').replace(':','_').replace('-','_').replace('.','_')

	def saveImage(self,img,iter):
		path=os.getcwd()
		print(path)
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

	def processImageDirectory(self,testPath="",resultPath=""):
		startingDirectory,testPath,originalTestImagesCpyPath,processedTestImagesClassPath,processedTestImagesPath,resultPath = self.formatDirectories(testPath,resultPath)		
		shutil.copytree(testPath, originalTestImagesCpyPath,dirs_exist_ok=True)
		os.chdir(originalTestImagesCpyPath)	
		images = []
		imgFound=0
		for image in os.listdir(originalTestImagesCpyPath):
			if image.endswith(".png") or image.endswith(".jpg"):
				print("image found: " + image)
				img = cv2.imread(image)
				images.append(img)
				imgFound+=1
			else:
				print("image format invalid: ")
		os.chdir(processedTestImagesClassPath) 
		print("Images found: " + str(imgFound))
		imgProcessed = 0
		for image in images:
			self.processImage(image)
			imgProcessed+=1
		print("Test images processed: " + str(imgProcessed))
		os.chdir(startingDirectory)
		return [processedTestImagesPath,resultPath]
	
	def formatDirectories(self,testPath="",resultPathParent=""):
		#Starting (original) directory
		startingDirectory= os.path.dirname(os.path.abspath(__file__))
		#test image "in" directory
		if not testPath:
			testPath = ".\\src\\data\\testData\\sampleNumbersOneImage"
			testPath = os.path.abspath(testPath)
		#results "out" directory
		todaynowstr = self.getTodayNow()
		if not resultPathParent:
			resultPathParent = ".\\.\\runs\\"
		resultPathParent = os.path.join(resultPathParent, todaynowstr)
		originalTestImagesCpyPath = os.path.join(resultPathParent, "originalTestImages\\")
		processedTestImagesPath = os.path.join(resultPathParent, "processedTestImages\\")
		processedTestImagesClassPath = os.path.join(processedTestImagesPath, "images\\")
		resultPath = os.path.join(resultPathParent,"results\\")
		#recursively makes directories
		os.makedirs(originalTestImagesCpyPath)
		os.makedirs(processedTestImagesClassPath)
		os.makedirs(resultPath)
		testPath = os.path.abspath(testPath)
		originalTestImagesCpyPath = os.path.abspath(originalTestImagesCpyPath)
		processedTestImagesClassPath = os.path.abspath(processedTestImagesClassPath)
		processedTestImagesPath = os.path.abspath(processedTestImagesPath)
		resultPath = os.path.abspath(resultPath)
		return startingDirectory,testPath,originalTestImagesCpyPath,processedTestImagesClassPath,processedTestImagesPath,resultPath
		