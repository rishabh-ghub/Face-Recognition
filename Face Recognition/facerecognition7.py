import cv2
import pickle
import numpy as np

cap=cv2.VideoCapture(0)
fd=cv2.CascadeClassifier(r'C:\Program Files\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
specs=0

labels={}
with open('labels.pickle','rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
    #pickle.dump(label_ids,f)

while(1):
    r,frame=cap.read()
    #print(r)
    gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    f=fd.detectMultiScale(gray)
    for [x,y,w,h] in f:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:
            #print(id_)
            #print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        if(specs==1):
            cv2.circle(frame,(int(x+w/3.5),int(y+h/2.5)),int(h/7),(50,0,0),3)
            cv2.circle(frame,(int(x+w-w/3.5),int(y+h/2.5)),int(h/7),(50,0,0),3)
            cv2.line(frame,(int(x+w/3.5+h/7),int(y+h/2.5)),(int(x+w-w/3.5-h/7),int(y+h/2.5)),(10,0,0),5)
            cv2.line(frame,(int(x+w/3.5+h/7),int(y+h-h/3.2)),(int(x+w-w/3.5-h/7),int(y+h-h/3.2)),(0,0,0),10)
        
        
        cv2.imshow('image',frame)
        
    k=cv2.waitKey(5)
    if(k==ord('s')):
        if(specs==1):
            specs=0
            pass
        elif(specs==0):
            specs=1
    if(k==ord('k')):
        cv2.imwrite('Capture.png',frame)
    
    if(k==ord('q')):        
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
