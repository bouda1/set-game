#!/usr/bin/python3.7

import cv2
import numpy as np

debug = False

def detect_shape(img, lst):
    tilda = cv2.imread('tilda.png', 0)
    patate = cv2.imread('patate.png', 0)
    losange = cv2.imread('losange.png', 0)
    height, width = losange.shape[:2]
    himg, wimg = img.shape[:2]

    is_tilda = 0
    is_patate = 0
    is_losange = 0
    for c in lst:
        img = np.zeros((himg, wimg, 1), np.uint8)
        img = cv2.drawContours(img, [c], 0, 255, 4)
        bounds = cv2.boundingRect(c)
        crop_img = img[bounds[1]:(bounds[1]+bounds[3]), bounds[0]:(bounds[0]+bounds[2])]
        crop_img = cv2.resize(crop_img, (width, height))

        score_losange = cv2.matchTemplate(crop_img, losange, cv2.TM_CCOEFF_NORMED)
        score_tilda = cv2.matchTemplate(crop_img, tilda, cv2.TM_CCOEFF_NORMED)
        score_patate = cv2.matchTemplate(crop_img, patate, cv2.TM_CCOEFF_NORMED)

        if score_losange > score_tilda and score_losange > score_patate:
            is_losange += 1
        elif score_tilda > score_losange and score_tilda > score_patate:
            is_tilda += 1
        else:
            is_patate += 1

    m = max(is_losange, is_patate, is_tilda)
    if m == is_tilda:
        return "tilda"
    if m == is_patate:
        return "patate"
    return "losange"


def detect_card(img):
    # Recognize the shape
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    blur = cv2.GaussianBlur(imgray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret3, thresh = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    lst = []

    for i in range(len(contours)):
        c = contours[i]

        area = cv2.contourArea(c)
        if 10000 < area < 20000:
            if hierarchy[i][3] >= 0:
                # There are children
                j = hierarchy[i][3]
                found = False
                while not found and j >= 0:
                    cc = contours[j]
                    area1 = cv2.contourArea(cc)
                    if 10000 < area1 < 20000:
                        found = True
                    j = hierarchy[j][1]
            if not found:
                # No parent with good size
                #epsilon = 0.8 * cv2.arcLength(c, True)
                #approx = cv2.approxPolyDP(c, epsilon, True)
                lst.append(c)
            nb = len(lst)
            shape = detect_shape(img, lst)

    if debug:
        new_img = cv2.drawContours(img, lst, -1, (0, 255, 255), 4)
        cv2.imshow("Shape detection", new_img)
    else:
        cv2.imshow("Shape detection", img)

    while True:
        if cv2.waitKey(1) == 27:
            break

    return nb, shape

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
        # The cards is transformed on a 2D plane.
        new_img = four_point_transform(img, c)

        # Recognize the shape
        imgray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        nb, shape = detect_card(new_img)

        #ret, thresh = cv2.threshold(imgray, 150, 255, 0)
        blur = cv2.GaussianBlur(imgray, (7, 7), 20)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #ret3, thresh = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        new_img = cv2.putText(new_img, str(nb) + " " + shape, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("mes cartes", new_img)
        while True:
            if cv2.waitKey(1) == 27:
                break


def read_cards(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # In a darker room
    #ret, thresh = cv2.threshold(imgray, 130, 255, 0)

    # In a lighter room
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)

    if debug:
        cv2.imshow("mes cartes", thresh)
        while True:
            if cv2.waitKey(1) == 27:
                break

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lst = []

    for i in range(len(contours)):
        c = contours[i]

        area = cv2.contourArea(c)
        if 2000 < area < 50000:
            # No parent with good size
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                print(area)
                lst.append(approx)

    if debug:
        new_img = cv2.drawContours(img, lst, -1, (0, 0, 255), 4)
        cv2.imshow("mes cartes", new_img)
    else:
        cv2.imshow("mes cartes", img)

    while True:
        if cv2.waitKey(1) == 27:
            break

    get_cards(img, lst)



def main():
    #show_webcam()
    img = cv2.imread('webcam2.png')
    read_cards(img)


if __name__ == '__main__':
    main()
