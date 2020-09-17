#!/usr/bin/python
import torch,torch.nn as nn
import time,cv2,os,sys,numpy as np,sys
from torchvision import transforms
from predictor.detect import detectorClass
import torchvision.models as models

"inits YOLOv3"
dc=detectorClass("predictor/cfg/yolo_v3.cfg","predictor/yolov3.weights")

"determines the touch's location"
def locator(posSum):
    if posSum<426:
        return "left"
    elif posSum<852:
        return "center"
    else:
        return "right"

devMode=True

"def of lower/upper bounds of overlay colours for HSV Red"
lower_red = np.array([160,100,230], dtype = "uint16")
upper_red = np.array([180,256,256], dtype = "uint16")

"def of lower/upper bounds of overlay colours for HSV Green"
lower_green = np.array([0,100,230], dtype = "uint16")
upper_green = np.array([100,256,256], dtype = "uint16")

"def of lower/upper bounds of overlay colours for RGB Gray"
lower_black = np.array([120,120,120], dtype = "uint16")
upper_black = np.array([220,220,220], dtype = "uint16")

"load model weights"
model = models.resnet34(pretrained=True).cuda()
model.load_state_dict(torch.load("gray-resnet34-sota.m"))
model.eval()

"if DataParallel, cuda, etc."
# model = torch.nn.DataParallel(model)
model = model.cuda()
model = model.eval()

"normalization defs for DL"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transformations=transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

"touch index for machine"
touchIndex={0:"counter-attack",1:"lunge",4:"preparation-to-attack"}

"incase YOLOv3 cannot detect 2 fencers, for the undetected fencer"
def oneHandler(posSum,position0,filetosave):
    print("touch@",posSum, file=filetosave)
    print("touch@",posSum)
    if posSum=="left":
        print("leftfencer counter-attack", file=filetosave)# or NaN
        print("rightfencer",touchIndex[position0[0]], file=filetosave)
        print("leftfencer counter-attack") # or NaN
        print("rightfencer",touchIndex[position0[0]])
    elif posSum=="center":
        print("leftfencer",touchIndex[position0[0]], file=filetosave)# or NaN
        print("rightfencer",touchIndex[position0[0]], file=filetosave)# or NaN
        print("leftfencer",touchIndex[position0[0]])
        print("rightfencer",touchIndex[position0[0]])
    elif posSum=="right":
        print("leftfencer",touchIndex[position0[0]], file=filetosave)
        print("rightfencer counter-attack", file=filetosave) # or NaN
        print("leftfencer",touchIndex[position0[0]])
        print("rightfencer counter-attack") # or NaN


"main function. insert file to save the match data and videoname to read"
def statisticalSave(videoNo,filetosave):
    cap = cv2.VideoCapture("./"+str(videoNo)+".mp4")

    "variable definition"
    jumpFrames=0
    first=True
    frameList=[]
    frameCounter=0

    initial=False

    "init 15 touch array to save the score images to perform image divisions."
    isFirst=True
    scoreSave=[]
    for adderr in range(16):
        scoreSave.append([[],[]])

    "current score counter"
    cL=0
    cR=0

    touch=False

    "runs video"
    while(cap.isOpened()):
        "resizes the given frame, crops to ROI, performs filtering to red/green seperately, adds them together"
        _, frame = cap.read()
        try:
            img = cv2.resize(frame,(1071, 604), interpolation = cv2.INTER_LANCZOS4) ##orig img.
            vFrame = img[514:564, 130:941]
            img_hsv = cv2.cvtColor(vFrame, cv2.COLOR_BGR2HSV)#toHSV
            img_hsvRed = cv2.inRange(img_hsv, lower_red, upper_red)
            img_hsvGreen = cv2.inRange(img_hsv, lower_green, upper_green)
            img_hsv = cv2.addWeighted(img_hsvRed,1,img_hsvGreen,1,0)

            "for results"
            touchLeft=cv2.cvtColor(vFrame[6:32,320:340], cv2.COLOR_BGR2GRAY)
            touchRight=cv2.cvtColor(vFrame[6:32,-340:-320], cv2.COLOR_BGR2GRAY)

            "for the first test touch, saves the scores."
            if initial:
                if isFirst:
                    isFirst=False
                    scoreSave[0][0].append(touchLeft)
                    scoreSave[0][1].append(touchRight)
                    saveL=scoreSave[0][0][0]
                    saveR=scoreSave[0][1][0]
                else:
                    "image division to observe change"
                    divL=touchLeft/saveL
                    divR=touchRight/saveR
                    mdL=np.min(divL)
                    mdR=np.min(divR)
                    if mdL<0.7 or mdR<0.70:
                        if mdL<0.7 and cL>0:
                            "checking if the score has been decreased"
                            divL=touchLeft/scoreSave[cL-1][0][0]
                            mdL=np.min(divL)
                            if mdL>0.8:
                                cL-=1
                                saveL=scoreSave[cL][0][0]
                            else:
                                print("point-to left", file=filetosave)
                                print("point-to left")
                                cL+=1
                                scoreSave[cL][0].append(touchLeft)
                                saveL=scoreSave[cL][0][0]
                        elif mdL<0.7 and cL==0:
                            print("point-to left", file=filetosave)
                            print("point-to left")
                            cL+=1
                            scoreSave[cL][0].append(touchLeft)
                            saveL=scoreSave[cL][0][0]

                        if mdR<0.7 and cR>0:
                            divR=touchRight/scoreSave[cR-1][1][0]
                            mdR=np.min(divR)
                            if mdR>0.8:
                                cR-=1
                                saveR=scoreSave[cR][1][0]
                            else:
                                print("point-to right", file=filetosave)
                                print("point-to right")
                                cR+=1
                                scoreSave[cR][1].append(touchRight)
                                saveR=scoreSave[cR][1][0]
                        elif mdR<0.7 and cR==0:
                            print("point-to right", file=filetosave)
                            print("point-to right")
                            cR+=1
                            scoreSave[cR][1].append(touchRight)
                            saveR=scoreSave[cR][1][0]

                        print(str(cL)+" "+str(cR), file=filetosave)
                        print(str(cL)+" "+str(cR))

            "adds the first frame directly, processes the onwards"
            if jumpFrames==0:
                frameList.append(cv2.countNonZero(img_hsv))
            else:
                "adds the frame, pops the last one"
                frameList.append(cv2.countNonZero(img_hsv))
                if not first:
                    frameList.pop(1)
                else:
                    first=False

                "where it detects, if there is difference"
                if (frameList[0]<1000 and frameList[1]>1000):

                    "next round"
                    print("----------------------------------", file=filetosave)
                    print("----------------------------------")

                    initial=True

                    "saves the image"
                    frameImageCounter=0
                    "to find position"
                    position=[]
                    "candidate detection"
                    twos=[]

                    "uses YOLOv3 of Andy Yun's to find humans"
                    DETECTED=dc.detect_cv2(frame)
                    "gets each boxed object"
                    for eacher in DETECTED[1]:
                        "checks if person"
                        if eacher[0]=="person":
                            "uses lower/upper bounds of RGB GRAY to detect if the box is of fencer's"
                            try:
                                crop_BWimg = cv2.inRange(DETECTED[0][eacher[2]:eacher[4],eacher[1]:eacher[3]], lower_black, upper_black)
                                y=np.count_nonzero(crop_BWimg)
                                "the filter threshold and save"
                                if y>4000:
                                    "resize and save the fencer"
                                    resizedFencer = cv2.resize(DETECTED[0][eacher[2]-30:eacher[4]+30,eacher[1]-30:eacher[3]+30],(224, 224), interpolation = cv2.INTER_LANCZOS4)

                                    "classify fencer"
                                    with torch.no_grad():
                                        "model specific color convert"
                                        "to be GRAY,HSV,Edge, or comment to leave it as RGB"
                                        resizedFencer=cv2.cvtColor(resizedFencer, cv2.COLOR_BGR2GRAY)
                                        resizedFencer=cv2.cvtColor(resizedFencer, cv2.COLOR_GRAY2BGR)

                                        "preprocessing"
                                        normalizedFencer=transformations(resizedFencer).cuda()

                                        outputs = model(normalizedFencer.unsqueeze_(0))
                                        __, predicted = torch.max(outputs.data, 1)

                                        "if ctr,lng,non; save apt places"
                                        if predicted==0:# green ctr
                                            frame=cv2.rectangle(frame,(eacher[1]-30,eacher[2]-30),(eacher[3]+30,eacher[4]+30),(0,255,0),3)
                                            position.append([0,(eacher[1]+eacher[3])/2,(eacher[2]+eacher[4])/2])
                                            touch=True

                                        elif predicted==1:# red lng
                                            frame=cv2.rectangle(frame,(eacher[1]-30,eacher[2]-30),(eacher[3]+30,eacher[4]+30),(0,0,255),3)
                                            position.append([1,(eacher[1]+eacher[3])/2,(eacher[2]+eacher[4])/2])
                                            touch=True
                                        elif predicted==2:
                                            frame=cv2.rectangle(frame,(eacher[1]-30,eacher[2]-30),(eacher[3]+30,eacher[4]+30),(0,255,255),3)
                                            "candidate frame"
                                            twos.append([2,(eacher[1]+eacher[3])/2,(eacher[2]+eacher[4])/2,eacher[1]-30,eacher[2]-30,eacher[3]+30,eacher[4]+30])
                                        elif predicted==3:
                                            frame=cv2.rectangle(frame,(eacher[1]-30,eacher[2]-30),(eacher[3]+30,eacher[4]+30),(0,255,255),3)
                                        elif predicted==4:
                                            frame=cv2.rectangle(frame,(eacher[1]-30,eacher[2]-30),(eacher[3]+30,eacher[4]+30),(255,0,0),3)
                                            position.append([4,(eacher[1]+eacher[3])/2,(eacher[2]+eacher[4])/2])
                                            touch=True

                                    frameImageCounter+=1
                            except Exception as e:
                                print(e)
                                pass
                                # print(e)
                                # raise

                    "location detector"
                    "if normally detected"
                    if len(position)>=2:
                        "if right fencer is left, change places"
                        if position[0][1]>position[1][1]:
                            position[1], position[0] = position[0], position[1]

                        posSum=(position[0][1]+position[1][1])/2

                        "give location"
                        print("touch@",locator(posSum), file=filetosave)
                        print("leftfencer",touchIndex[position[0][0]], file=filetosave)
                        print("rightfencer",touchIndex[position[1][0]], file=filetosave)
                        print("touch@",locator(posSum))
                        print("leftfencer",touchIndex[position[0][0]])
                        print("rightfencer",touchIndex[position[1][0]])

                        "if 1 misclassification"
                    elif len(position)==1:
                        "if any candidate fencer"
                        if len(twos)>0:
                            temp=720
                            ind=0
                            "find the candidate whose y-coord is the closest one"
                            for eachtwo in range(len(twos)):
                                tmp=abs(position[0][2]-twos[eachtwo][2])
                                if temp>tmp:
                                    ind=eachtwo
                                    temp=tmp

                            "if the candidate is in range"
                            if temp<80:
                                position.append(twos[ind])
                                position[1][1]=1
                                "reallign the left-right fencer"
                                isNewLeft=False
                                if position[0][1]>position[1][1]:
                                    position[1], position[0] = position[0], position[1]
                                    isNewLeft=True
                                posSum=(position[0][1]+position[1][1])/2

                                "position assigner"
                                "left"
                                if posSum<426:
                                    print("touch@ left", file=filetosave)
                                    print("touch@ left")
                                    if isNewLeft:
                                        position[0][0]=0
                                        frame=cv2.rectangle(frame,(twos[ind][3],twos[ind][4]),(twos[ind][5],twos[ind][6]),(0,255,0),3)
                                    else:
                                        position[0][0]=1
                                        frame=cv2.rectangle(frame,(twos[ind][3],twos[ind][4]),(twos[ind][5],twos[ind][6]),(0,0,255),3)
                                    "center"
                                elif posSum<852:
                                    print("touch@ center", file=filetosave)
                                    print("touch@ center")
                                    frame=cv2.rectangle(frame,(twos[ind][3],twos[ind][4]),(twos[ind][5],twos[ind][6]),(0,0,255),3)
                                    "right"
                                else:
                                    print("touch@ right", file=filetosave)
                                    print("touch@ right")
                                    if not isNewLeft:
                                        position[0][0]=0
                                        frame=cv2.rectangle(frame,(twos[ind][3],twos[ind][4]),(twos[ind][5],twos[ind][6]),(0,255,0),3)
                                    else:
                                        position[0][0]=1
                                        frame=cv2.rectangle(frame,(twos[ind][3],twos[ind][4]),(twos[ind][5],twos[ind][6]),(0,0,255),3)
                                try:
                                    print("leftfencer",touchIndex[position[0][0]], file=filetosave)
                                    print("rightfencer",touchIndex[position[1][0]], file=filetosave)
                                    print("leftfencer",touchIndex[position[0][0]])
                                    print("rightfencer",touchIndex[position[1][0]])
                                except:
                                    # print(position, file=filetosave)
                                    # print(position)
                                    pass
                            else:
                                "if not in range"
                                oneHandler(locator(position[0][1]),position[0],filetosave)
                        else:
                            "twos are not there"
                            oneHandler(locator(position[0][1]),position[0],filetosave)

                    frameCounter+=1

                if devMode:
                    cv2.imshow("Video",frame)
                    if not touch:
                        cv2.waitKey(1)
                    else:
                        cv2.waitKey(500)
                        touch=False
                frameList.pop(0)
                jumpFrames=-1
            jumpFrames+=1
        except Exception as e:
            # print(e)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    with open(sys.argv[1]+'.txt', 'w') as f:
        statisticalSave(sys.argv[1],f)
