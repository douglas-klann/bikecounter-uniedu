import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('video1.mov')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

#variavel pro loop
count=0

#inicia o tracking
tracker=Tracker()

#coordenadas linha esquerda
cy1=40
cx1=350

#coordenadas linha direita
cy2=470
cx2=650

#offset para detecção do centro na linha
offset=15

#Dicionarios de IDs e Coordenada X que passaram nas linhas
bk_in = {}
bk_out = {}

#Lista de IDs que contabilizaram nas linhas
counter_in = []
counter_out = []

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
             
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'bicycle' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        #Contabiliza as bicicletas entrando
        if cx2 < (cx+offset) and cx2 > (cx-offset):
            bk_in[id] = cx
        if id in bk_in:
            if cx1 < (cx+offset) and cx1 > (cx-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                counter_in.append(id)

        #Contabiliza as bicicletas saindo
        if cx1 < (cx+offset) and cx1 > (cx-offset):
            bk_out[id] = cx
        if id in bk_out:
            if cx2 < (cx+offset) and cx2 > (cx-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                counter_out.append(id)
           
    #Desenha as duas linhas
    cv2.line(frame,(cx1, cy1),(cx1, cy2),(255,255,255),1)
    cv2.line(frame,(cx2, cy1),(cx2, cy2),(255,255,255),1)

    #Mostra a quantidade de bicicletas que entraram
    print(bk_in)
    count_in = len(counter_in)
    cv2.putText(frame,("Entrou: ") + str(count_in),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    #Mostra a quantidade de bicicletas que saíram
    print(bk_out)
    count_out = len(counter_out)
    cv2.putText(frame,("Saiu: ") + str(count_out),(60,70),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    #Exibe o vídeo
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()