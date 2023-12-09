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
		print("Contour images saved: " + str(iter))


	def processTestImageDirectory(self,testPath="",resultPath=""):		
		# Argument: testPath   = user-entered, validated path to directory of test images
		# Argument: resultPath = user-entered, validated path to directory to save result images

		# defined in formatDirectories()
		homePath,testPath,ogTestPath,pccdTestImgPath,pccdTestImgClssPath,resultPath = self.formatDirectories(testPath,resultPath)		
		
		shutil.copytree(testPath,ogTestPath,dirs_exist_ok=True)	
		os.chdir(ogTestPath)
		
		# Sort out images of unacceptable filetypes
		images = []
		imgFound=0
		for image in os.listdir(ogTestPath):
			if image.endswith(".png") or image.endswith(".jpg"):
				print("Test image found: " + image)
				img = cv2.imread(image)
				images.append(img)
				imgFound+=1
			else:
				print("Test image format invalid. jpg and png only.")
		
		# Process valid images
		os.chdir(pccdTestImgClssPath) 
		print("Test images found: " + str(imgFound))
		imgProcessed = 0
		for image in images:
			self.processImage(image)
			imgProcessed+=1
		print("Test images processed: " + str(imgProcessed))
		os.chdir(homePath)
		return [pccdTestImgPath,resultPath]
	
	def formatDirectories(self,testPath="",resultPathParent=""):
		# homePath             = absolute path of this file
		# ogTestPath           = absolute path of where orginal test images are copied to in results folder
		# pccdTestImgPath      = absolute path of processed test images
		# pccdTestImgClssPath  = absolute path of class file for processed test images 
		# resultPath           = absolute path of directory where the result plots and csv will be stored
		
		# |-homepath (..\HandWritingTracker)
        # |----runs
        # |-------datetime
        # |-----------originalTestImages
        # |-----------processedTestImages
        # |---------------images
        # |-----------results
        # |----src
        # |------data
        # |-----------sampleNumbersOneImage		

		homePath= os.path.dirname(os.path.abspath(__file__))
		
		# testPath   = user-entered, validated path to directory of test images
		if not testPath:
			testPath = ".\\src\\data\\testData\\sampleNumbersOneImage"
		testPath = os.path.abspath(testPath)
		
		# directory to uniquly identify this run
		todaynowstr = self.getTodayNow()

		# resultPathParent = parent folder for this run
		if not resultPathParent:
			resultPathParent = ".\\.\\runs\\"		
		resultPathParent = os.path.join(resultPathParent, todaynowstr)
		
		# subfolders for this run: 
		# original test images are copied to this folder
		ogTestPath = os.path.join(resultPathParent, "originalTestImages\\")
		# processed test images are stored in class folder in this folder
		pccdTestImgPath = os.path.join(resultPathParent, "processedTestImages\\")
		pccdTestImgClssPath = os.path.join(pccdTestImgPath, "images\\")
		# result data is stored in this folder
		resultPath = os.path.join(resultPathParent,"results\\")
		
		# recursively makes directories
		os.makedirs(ogTestPath)
		os.makedirs(pccdTestImgClssPath)
		os.makedirs(resultPath)
		
		# return absolute paths
		testPath = os.path.abspath(testPath)
		ogTestPath = os.path.abspath(ogTestPath)
		pccdTestImgPath = os.path.abspath(pccdTestImgPath)
		pccdTestImgClssPath = os.path.abspath(pccdTestImgClssPath)
		resultPath = os.path.abspath(resultPath)
		
		return homePath,testPath,ogTestPath,pccdTestImgPath,pccdTestImgClssPath,resultPath
		