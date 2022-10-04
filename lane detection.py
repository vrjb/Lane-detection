import cv2 as cv
import numpy as np

def draw_the_line(image, lines):
    line_image= np.zeros((image.shape[0],image.shape[1],3),'uint8')
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(0,255,0), 3)
    image_with_lines= cv.addWeighted(image,0.8,line_image,1,0)
    return image_with_lines

def region_of_interest(image,region_point):
    mask= np.zeros_like(image)
    cv.fillPoly(mask,region_point, 255)
    masked_img=cv.bitwise_and(image, mask)
    return masked_img

def detect_lane(image):
    height, width =image.shape[0:2]
    gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    blur= cv.GaussianBlur(gray_image,(5,5),1)
    canny= cv.Canny(blur,50,150)
    region_of_interest_vertices = np.array([[(0, height), (width / 2, height * 0.65), (width, height)]],np.int32)
    cropped_image= region_of_interest(canny,region_of_interest_vertices)
    lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40,
                           maxLineGap=150
                           )

    image_with_lines = draw_the_line(image, lines)

    return image_with_lines


video= cv.VideoCapture(r"C:\Users\captain jacksparrow\Desktop\computer_vision_course_materials\lane_detection_video.mp4")

while video.isOpened():
    is_grabbed, frame = video.read()
    if not is_grabbed:
        break
    frames = detect_lane(frame)
    cv.imshow('lane', frames)
    k= cv.waitKey(20)
    if k & 0xFF == ord('q'):
        break


video.release()
cv.destroyAllWindows()


