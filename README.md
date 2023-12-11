To begin, copy the repository using the following git commad: 

git clone https://github.com/aandrade1223/HandWritingTracker.git

Before running this project, ensure you are either passing in a path to a directory where test images are stored as jpegs or pngs or copy and paste test images in form of jpegs and pngs to src/data/testData/testImages. 

To run this project, change directory into HandwritingTracker/src. Run with command "python main.py"

There are 9 optional arguments to this program:
1. -ct or --confidencethreshold : 0 to 1 probability that determines an acceptable prediction confidence of an image classification. Default 0.8
2. -vt or --variancethreshold : acceptable logits variance threshold. Default 0.75
3. -tr or --trainpath : path to directory of training dataset of images
4. -v  or --validpath : path to directory of validation dataset of images
5. -te or --testpath : path to directory of test images
6. -r  or --resultpath : path to directory where results are stored
7. -c  or --buildCSV : flag to build csv file of results. Default True
8. -p  or --plotResults : flag to plot images with results. Default True
9. -o  or --overrideExits : flag to override any early exits. Default False

Results of image classifications are stored in HandWritingTracker/runs/{datetime}/results. 
