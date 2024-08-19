import cv2
from PIL import Image, ImageTk
import os
import pickle
import numpy as np
import ttkbootstrap as ttk
# from subprocess import run

#ttkbootstrap
window = ttk.Window(themename='vapor')
window.title("Application de Reconnaissance Faciale")
window.geometry("1080x720")
L5 = ttk.Label(window, text = '' )
L5.pack(side="right", fill="y")
L6 = ttk.Label(window, text = '' )
L6.pack(side="left", fill="y")
f1 = ttk.Frame(window)
f2 = ttk.Frame(window)
f3= ttk.Frame(window)
f4= ttk.Frame(window)
f5= ttk.Frame(window)
f1.pack(pady=10)
f2.pack(pady=10)
f3.pack(pady=10)
f4.pack()
f5.pack(side=ttk.LEFT)

L1 = ttk.Label(f1)
L1.pack()
pb = ttk.Progressbar(f1, orient='horizontal', mode='determinate', length=200, bootstyle='primary-tripped')
pb.pack(pady=10)
L2 = ttk.Label(f3, text='Entrer le nom de la personne:',font= ("Courier New",10))
L2.pack(pady=5, fill= ttk.X)
entry = ttk.Entry(f3)
entry.pack(pady=10, fill= ttk.X)
L3 = ttk.Label(f4, text='', font= ("Courier New",20), bootstyle='info')
L3.pack(pady=20)
L4 = ttk.Label(f4, text = '',font= ("Courier New",10) )
L4.pack()


#Configuration path
BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR,"images")
script_path = os.path.join(BASE_DIR, "reconnaissance_faciale.py")

#Lbph algorithm
recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=500)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

def delete_all():
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            path = os.path.join(root, file)
            os.remove(path)
        if root != img_dir:  
        	os.rmdir(root)
         
def delete_one():
    dirId = int(entry_del.get())
    all_dir = {}
    id = 0
    for root, dirs, files in os.walk(img_dir):
        all_dir[id] = os.path.basename(root)
        id+=1
    del_path = os.path.join(img_dir, all_dir[dirId+1] )   
    for root, dirs, files in os.walk(del_path):
        for file in files:
            path = os.path.join(root, file)
            os.remove(path)
        os.rmdir(root)
        L4['text'] = "{} a été supprimé :(".format(all_dir[dirId+1]) 
    
  
        
def apprentissage():
	image_dir = os.path.join(BASE_DIR, "images")
	save_path = os.path.join(BASE_DIR, "modele_entraine.yml")
	label_path = os.path.join(BASE_DIR,"labels.pickle")  
 
	current_id = 0
	label_ids = {}
	y_labels = []
	x_train = []
    
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg"):
				path = os.path.join(root, file)
				label = os.path.basename(root).replace(" ","-").lower()
				print(path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id +=1
		
				id_ = label_ids[label]
				image_array = cv2.imread(path)
				gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
				x_train.append(gray_img)
				y_labels.append(id_)
     
	with open(label_path, "wb") as f:
		pickle.dump(label_ids, f)
	L4['text']='{} personne(s) répertoriée(s): {}'.format(len(label_ids),label_ids)
	recognizer.train(x_train, np.array(y_labels))
	recognizer.save(save_path)
 #fin
    
 
def capture():
    pb['value']=0
    L3['text']=""
    dir_name = entry.get()
    path = os.path.join(img_dir, dir_name)
    os.mkdir(path)
  
    cap = cv2.VideoCapture(0)
    img_count = 0
    color = (0, 255, 0)
    
    while img_count < 200:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(480,296))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Début
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors= 5, minSize=(50,50))
        for (x,y,w,h) in faces:
            roi = cv2.resize(img_gray[y:y+h, x:x+w], (200,200))
            img_item = "capture-{}-{}.png".format(dir_name, img_count)
            write_path = os.path.join(path, img_item)
            cv2.imwrite(write_path,roi)
            #fin
            img_count+=1
            pb['value']+=0.5
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
                
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        L1['image'] = frame
        window.update()
    entry.delete(0, ttk.END) 
    L3['text']="Capture terminée :)"
    
    
    
capture_button = ttk.Button(f3, text="Capture", bootstyle= 'success-outline', command=capture)
learn_button = ttk.Button(f3, text="Apprentissage",bootstyle= 'outline', command=apprentissage)
entry_label = ttk.Label(f3, text='Entrer le numéro de la personne à supprimer:',font= ("Courier New",10))
entry_del = ttk.Entry(f3)
del_one_button = ttk.Button(f3, text="Supprimer",bootstyle= 'danger-outline', command=delete_one, width=15)
del_button = ttk.Button(f3, text="Supprimer Tout",bootstyle= 'danger-outline', command=delete_all, width=15)
capture_button.pack( pady= 10, fill= ttk.X)
learn_button.pack(pady=10,fill= ttk.X)
entry_label.pack(pady=5,fill= ttk.X)
entry_del.pack(pady=5,fill= ttk.X)
del_one_button.pack(pady=10)
del_button.pack(pady=10)
window.mainloop()
