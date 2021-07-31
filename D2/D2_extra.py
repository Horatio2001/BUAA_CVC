import cv2
import numpy as np
import matplotlib
from PIL import Image

# img1 = cv2.resize(img1,dsize=(600,400))
# image1 = img1.copy()
def My_fast(image1):
    fast = cv2.FastFeatureDetector_create(threshold=50)
    keypoints1 = fast.detect(image1, None)
    # 在图像上绘制关键点
    image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255), \
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print('show')
    cv2.imshow('fast_keypoints1', image1)
    cv2.waitKey(0)


def My_orb(image):
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = gray1.copy()
    orb = cv2.ORB_create(128)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255), \
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('orb_keypoints1', image1)
    cv2.waitKey(0)


def My_sift(image1,image2):
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img1 = gray.copy()
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    img2 = gray.copy()
    sift = cv2.xfeatures2d.SIFT_create(400)
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)
    img1 = cv2.drawKeypoints(image=img1, keypoints=keypoints1, outImage=img1, color=(255, 0, 255), \
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(image=img2, keypoints=keypoints1, outImage=img2, color=(255, 0, 255), \
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matcher = cv2.FlannBasedMatcher()
    matchePoints = matcher.match(descriptor1, descriptor2)
    minMatch = 1
    maxMatch = 0
    for i in range(len(matchePoints)):
        if minMatch > matchePoints[i].distance:
            minMatch = matchePoints[i].distance
        if maxMatch < matchePoints[i].distance:
            maxMatch = matchePoints[i].distance
    goodMatchePoints = []
    for i in range(len(matchePoints)):
        if matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 3:
            goodMatchePoints.append(matchePoints[i])
    outImg = None
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, outImg, matchColor=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow('matche', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def My_harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 3, 23, 0.04)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow(' ', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img1 = cv2.imread('pics3/turningtorso1.jpg')
    cv2.imshow(' ',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()