import cv2
import helper_functions as hf
import os
import shutil


def saveImage(img):
    todaynowstr = hf.getTodayNowStr()
    filename = "testContour_" + todaynowstr
    i = 0
    while os.path.exists(f"{filename}{i}.png"):
        i += 1
    filename = f"{filename}{i}.png"
    cv2.imwrite(filename, img)
    # print("image saved: " + filename)


def processImage(image):
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8
    )

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh, rect_kernel, iterations=5)

    # Invert image
    img_invert = cv2.bitwise_not(image)

    # Finding contours
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Creating a copy of image
    im2 = img_invert.copy()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    count = 0
    for iter, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y : y + h, x : x + w]

        # resize image
        cropped = cv2.resize(cropped, (28, 28))
        saveImage(cropped)
        count = iter + 1
    # if count==0:
    # 	print("No contors were found in image.")
    # else:
    # 	print("Contour images saved: " + str(count))
    return count


def processTestImageDirectory(
    cwdPath, copyFromTestPath, copyToTestPath, pccdTestImgClssPath, overrideExit
):

    # If there are no test images, exit early
    if not os.listdir(copyFromTestPath):
        print("No test images available.")
        if not overrideExit:
            print("Exiting early...")
            exit()
    else:
        shutil.copytree(copyFromTestPath, copyToTestPath, dirs_exist_ok=True)
    os.chdir(copyToTestPath)

    # Sort out images of unacceptable filetypes
    images = []
    imgFound = 0
    for image in os.listdir(copyToTestPath):
        if image.endswith(".png") or image.endswith(".jpg"):
            print("Test image found: " + image)
            img = cv2.imread(image)
            images.append(img)
            imgFound += 1
        else:
            print("Test image format invalid. jpg and png only.")

    # Process valid images
    os.chdir(pccdTestImgClssPath)
    print("Test images found: " + str(imgFound))
    imgProcessed = 0
    contorsFound = 0
    for image in images:
        contorsFound += processImage(image)
        imgProcessed += 1
    print("Test images processed: " + str(imgProcessed))
    print("Contours Found: " + str(contorsFound))
    os.chdir(cwdPath)