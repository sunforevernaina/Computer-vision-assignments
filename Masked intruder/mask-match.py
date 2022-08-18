import cv2
import os
import argparse
parser = argparse.ArgumentParser(description= "Giving Masked Images")
parser.add_argument('-i',type = str, help = "Input the MaskLow Image")
parser.add_argument('-j',type = str, help = "Input the MaskMiddle Image")
args = parser.parse_args()

def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2) # finding matches
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])  # good matches
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = [] # again filtering the matches 
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

imgs = [str(args.i), str(args.j)] # list of masked images
#img1 = cv2.imread("../replicate/capturedImages/maskMiddle4.jpg")  # queryImage
#img1 = cv2.resize(src, (250,180))
path1 = "../replicate/capturedImages" # path for query images
path2 = "../replicate/database_image"
dir_list = os.listdir(path2)
scores={}
retscores={}

for paths in imgs:
    img1 = cv2.imread(os.path.join(path1,paths))  # queryImage 
    for i in dir_list:
    
        img2 = cv2.imread(os.path.join(path2,i)) # trainImage
        #img2 = cv2.resize(dst, (250,180))
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        bf = cv2.BFMatcher()
    
        good = calculateMatches(des1,des2) # find good matches
        
        number_keypoints = 0
        if len(kp1) <= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)
        
        score = len(good)/ number_keypoints #computing score
        scores[i] = score # appending score of each image into a dictionary
        
        if score>=0.005:
            retscores[i] = score  # appending score of each image satisfying the threshold into dictionary

    if "Middle.jpg" in retscores.keys():
        print("Intruder {} is in database".format(paths))
        print("Score of Middle.jpg:",retscores['Middle.jpg'])   
    else:
        print("Intruder {} is not in database".format(paths))
        print("Score of Middle.jpg:",scores['Middle.jpg'])

