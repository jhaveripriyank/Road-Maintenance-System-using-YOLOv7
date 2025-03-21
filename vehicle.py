import cv2
from cv2 import blur
import numpy as np

#Webcamera
cap  = cv2.VideoCapture('video.mp4')
count_line_position = 550
min_width_rect = 80
min_height_rect = 80
#Algorithm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
def centre_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 7 #Pixel Error Margin
counter = 0
ltv = 0
htv = 0
weight = 0

while True:
    ret,frame1 = cap.read()

    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5,)
    #We shall now apply the above algorithm on every frame
    img_sub = algo.apply(blur)

    #Dilation
    dil = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dildata = cv2.morphologyEx(dil,cv2.MORPH_CLOSE,kernel)
    dildata = cv2.morphologyEx(dildata,cv2.MORPH_CLOSE,kernel)
    contour,h = cv2.findContours(dildata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1100,count_line_position),(255,255,0),3)

    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
                continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle "+str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        centre  = centre_handle(x,y,w,h)
        detect.append(centre)
        cv2.circle(frame1,centre,4,(0,0,255),-1)



        for(x,y) in detect:
            if(y<count_line_position+offset) and (y>count_line_position-offset):
                counter+=1
                area = cv2.contourArea(c)
                print(area)
                cv2.line(frame1,(25,count_line_position),(1100,count_line_position),(0,127,255),3)
                detect.remove((x,y))
                if area>500 and area<15000:
                    ltv = ltv+1
                elif area>15000 and area<125000:
                    htv = htv+1
                print("Vehicle Count: "+str(counter))
               # weight=(counter*1200)
               # print("Weight: "+str(weight))
                
                

    
    cv2.putText(frame1,"Vehicle Count: "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame1,"LTV: "+str(ltv),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame1,"HTV: "+str(htv),(50,170),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame1,"HTV Weight: "+str((htv*33000)/80000),(50,270),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Detector',dildata) 
    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()



