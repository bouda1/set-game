#!/usr/bin/python3.7

import cv2
import numpy as np


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

        cv2.imwrite('webcam3.png', img)
    cv2.destroyAllWindows()


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


def order_points(quad):
    pts = np.zeros((4, 2), dtype="float32")
    pts[0] = quad[0]
    pts[1] = quad[1]
    pts[2] = quad[2]
    pts[3] = quad[3]
    s = np.sum(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    rect[0] = quad[np.argmin(s)]
    rect[2] = quad[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    height = 450
    width = 300
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    # return the warped image
    return warped


def get_cards(img, contours):
    for c in contours:
        new_img = four_point_transform(img, c)
        cv2.imshow("mes cartes", new_img)
        while True:
            if cv2.waitKey(1) == 27:
                break


def read_cards(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 130, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lst = []

    for i in range(len(contours)):
        c = contours[i]

        area = cv2.contourArea(c)
        if 9000 < area < 50000:
            # No parent with good size
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                print(area)
                lst.append(approx)

    #new_img = cv2.drawContours(img, lst, -1, (0, 0, 255), 4)

    cv2.imshow("mes cartes", img)
    while True:
        if cv2.waitKey(1) == 27:
            break

    get_cards(img, lst)



def main():
    img = cv2.imread('webcam1.png')
    read_cards(img)


if __name__ == '__main__':
    main()
