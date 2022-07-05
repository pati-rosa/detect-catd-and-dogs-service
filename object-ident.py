import cv2
from datetime import datetime
import requests
#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pati/Documents/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/Users/patriciarosa/Documents/tcc/detect-catd-and-dogs-service/object-ident.py/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/Users/patriciarosa/Documents/tcc/detect-catd-and-dogs-service/object-ident.py/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    #cap.set(10,70)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    codec = cv2.VideoWriter_fourcc('m','p','4','v')
    print(codec)
    
    videoWriter = cv2.VideoWriter('captured7.avi', codec, 15, size)
    print(videoWriter)

    gravando = False

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.40,0.2, objects=['cat','dog' ])
        
        if objectInfo != [] :
            gravando = True
            videoWriter.write(img)
            
        print(gravando)

        if objectInfo == [] and gravando == True :
            videoWriter.release()
            req = requests.post("http://192.168.18.2:8080/videos/upload",
                files = { "foto": open("captured7.avi", "rb") }   
                )
            gravando =  False
            print(gravando)

            
        cv2.imshow("Output",img)
        cv2.waitKey(1)
