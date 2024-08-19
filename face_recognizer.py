
import cv2
import pickle
import os
import ttkbootstrap as ttk
from PIL import Image, ImageTk

window = ttk.Window(themename='vapor')
window.title("Application de Reconnaissance Faciale")
window.geometry("1024x720")
l4 = ttk.Label(window, text = '' )
l4.pack(side="right", fill="y")
l5 = ttk.Label(window, text = '' )
l5.pack(side="left", fill="y")
f1 = ttk.Frame(window)
f2 = ttk.Frame(window)
f3 = ttk.Frame(window)
f4 = ttk.Frame(window)
f1.pack(pady=10)
f2.pack()
f3.pack()
f4.pack()
l1 = ttk.Label(f1)
l1.pack()
l2= ttk.Label(f2, font=("Courier New", 15), bootstyle='primary')
l2.pack()
l3= ttk.Label(f2, font=("Courier New", 15), bootstyle='danger')
l3.pack()
BASE_DIR =  os.path.dirname(os.path.abspath(__file__))

def recognizer(cam):             
    cap = cv2.VideoCapture(cam)
    BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
    yml_path = os.path.join(BASE_DIR,"modele_entraine.yml")    
    label_path = os.path.join(BASE_DIR,"labels.pickle")            
    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    recognizer=cv2.face.LBPHFaceRecognizer_create(threshold = 500)
    recognizer.read(yml_path)
    frame_count = 0
    random_count = 0
    color_info=(255, 255, 255)
    color_ko=(255,0, 0)
    color_ok=(0, 255, 0)
    work = True
    with open(label_path, "rb") as f:
        og_labels=pickle.load(f)
        labels={v:k for k, v in og_labels.items()}
        print(labels)

    while work:
        ret, frame=cap.read()
        frame = cv2.resize(frame, (400,296))
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tickmark=cv2.getTickCount()
        #début
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5, minSize=(50,50))
        
        for (x, y, w, h) in faces:
            roi_gray= cv2.resize(gray[y:y+h, x:x+w], (200,200))
            id_, conf = recognizer.predict(roi_gray)
            #fin
            if conf < 40:
                color=color_ok
                name=labels[id_]
                frame_count+=1
                
                if frame_count >= 50:
                    l2['text'] = "Bienvenu {} !!! :)".format(name)
                    l3['text']=""
                    work = False
                
            else:
                color=color_ko
                name="Inconnu"
                if random_count == 100:
                    l3['text'] = "Désolé,je ne vous connais pas!:( "
                    l2['text'] = ""
                    work = False
                 
                else:
                    work = True
                random_count += 1
                
                
            label=name+" "+'{:5.2f}'.format(conf)
            cv2.putText(frame2, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
            cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 2)
        fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark) 
        cv2.putText(frame2, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
        frame = ImageTk.PhotoImage(Image.fromarray(frame2))
        l1['image'] = frame
        window.update()
        
        
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break
        if key==ord('a'):
            for cpt in range(100):
                ret, frame=cap.read()
 
#ttkbootstrap command
l6 = ttk.Label(f4, text = '',font= ("Courier New",10) )
pwd = ttk.Entry(f4)
button1 = ttk.Button(f3, text="PC_Webcam", bootstyle='success', command= lambda: recognizer(0))
button1.pack(pady=10, fill = ttk.X)
pwd.pack(pady=10,fill = ttk.X)
l6.pack(pady=10,fill = ttk.X)
window.mainloop()
