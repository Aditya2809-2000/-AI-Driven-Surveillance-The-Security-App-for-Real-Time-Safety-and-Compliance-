import bs4
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from itertools import combinations
import math
from imutils.video import VideoStream
import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tkinter import *
from tkinter.messagebox import *
from tkinter.scrolledtext import *
from PIL import ImageTk, Image  




protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
#detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def f1():
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "LPC: {}".format(lpc_count)
        opc_txt = "OPC: {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def f2():
    proto_txt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)
    mask_detector = load_model('mask_detector.model')
    cap = cv2.VideoCapture("mask.mp4")
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

        face_detector.setInput(blob)
        detections = face_detector.forward()

        faces = []
        bbox = []
        results = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                bbox.append((startX, startY, endX, endY))

        if len(faces) > 0:
            results = mask_detector.predict(faces)

        for (face_box, result) in zip(bbox, results):
            (startX, startY, endX, endY) = face_box
            (mask, withoutMask) = result

            label = ""
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) 
        if key == ord('q'):
                break

    cv2.destroyAllWindows()



def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def f3():
    cap = cv2.VideoCapture("testvideo2.mp4")

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        centroid_dict = dict()
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)


            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            # text = "ID: {}".format(objectId)
            # cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)


        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()



def f4():
	try:
		wa = "https://ipinfo.io/"
		res = requests.get(wa)

		data = res.json()
		city_name = data['city']
		
	
		loc = data['loc']
		
	
		a1 = "http://api.openweathermap.org/data/2.5/weather?units=metric"
		a2 = "&q=" + city_name
		a3 = "&appid=" +  "c6e315d09197cec231495138183954bd"

		wa = a1 + a2 + a3
		res = requests.get(wa)
	

		data = res.json()
		

		main = data['main']
		temp = main['temp']
	

		wa = "https://www.brainyquote.com/quote_of_the_day"
		res = requests.get(wa)
	
		
		data = bs4.BeautifulSoup(res.text, 'html.parser')

		info = data.find('img', {'class':'p-qotd'})	

		
		msg = info['alt']
		
		
		main_window_lbl1['text'] = 'Location:', city_name,   'Temp:'   , temp, 'degree celcius'
		main_window_lbl2['text'] = 'QOTD:' +  str(msg)	
		
	except Exception as e:
		print("issue ", e)


def f5():
	excel_window.deiconify()
	main_window.withdraw()
	data = pd.read_csv("salary_1.csv")
	print(data)
	print(data.isnull().sum())
		
	feature = data[["exp"]]
	target = data[["sal"]]

	model = LinearRegression()
	model.fit(feature, target)

	exp = excel_window_ent_exp.get()
	sal = model.predict([[exp]])
	showinfo("Number of persons ", sal)
	
	excel_window_ent_time.delete(0, END)
	excel_window_ent_time.focus()

def f6():
	main_window.deiconify()
	excel_window.withdraw()

'''def time():
string = strftime('%H:%M:%S %p')
main_window_lbl3.config(text = string)
mian_window_lbl3.after(1000, time)'''
 
	

splash = Tk()
splash.after(2000, splash.destroy)
splash.wm_attributes('-fullscreen', 'true')
msg = Label(splash, text="People Surveillance system", font=('Calibri', 50,'italic'), fg='black')
msg.pack()
splash.mainloop()





main_window = Tk()
main_window.title("SAS Security app")
main_window.geometry("430x500+450+100")
main_window.configure(bg="gray80")
photo = PhotoImage(file = "a1.png")
main_window.iconphoto(False, photo)




dwnd = PhotoImage(file='download.png')
main_window_lbl_2 = Label(main_window, image=dwnd, text="Computervison Desktop App",font=('Arial', 20, 'underline'),bd=2, relief='solid', width=200)
main_window_btn_counter = Button(main_window, text="PERSON COUNTER CAMERA", font=('Arial', 20, 'bold'),bd=2, relief='solid', width=25, command=f1)
main_window_btn_md = Button(main_window, text="MASK DETECTION CAMERA", font=('Arial', 20, 'bold'),bd=2, relief='solid', width=25, command=f2)
main_window_btn_sd = Button(main_window, text="SOCIAL DISTANCING CAMERA", font=('Arial', 20, 'bold'),bd=2, relief='solid', width=25, command=f3)
main_window_btn_ed = Button(main_window, text="DATA", font=('Arial', 20, 'bold'),bd=2, relief='solid', width=25, command=f5)
main_window_lbl1 = Label(main_window, font=('Arial', 20, 'bold' ), bd=2, relief='solid' )
main_window_lbl2 = Label(main_window, font=('Arial', 20, 'bold' ), bd=2, relief='solid')
# main_window__lbl3 = Label(main_window, font = ('Arial', 10, 'bold'),bg = 'black',fg = 'white')



main_window_lbl_2.grid(row = 0, column=0, pady=5)
main_window_btn_counter.grid(row = 1, column=0, pady=5)
main_window_btn_md.grid(row = 2, column=0, padx=5, pady=5)
main_window_btn_sd.grid(row = 3, column=0, padx=5, pady=5)
main_window_btn_ed.grid(row = 4, column=0, padx=5, pady=5)
main_window_lbl1.grid(row = 5, column=0, padx=5, pady=5)
main_window_lbl2.grid(row = 6, column=0, padx=5, pady=5)
# main_window_lbl3.grid(row = 0, column=3, padx=5, pady=5)
# time()
f4()




excel_window = Toplevel(main_window)
excel_window.title("People Data")
excel_window.geometry("430x500+450+100")
excel_window.configure(bg="gray80")
excel_window.geometry("500x500+400+100")


excel_window_lbl_time = Label(excel_window, text="enter time:", font=('Arial', 20, 'bold'), bd=2, relief='solid')
excel_window_ent_exp = Entry(excel_window, bd=5, font=('Arial', 20, 'bold'), relief='solid')
excel_window_btn_result = Button(excel_window, text="Result", width=10, font=('Arial', 20, 'bold'), bd=2, relief='solid',command=f5) 
excel_window_btn_back = Button(excel_window, text="Back", width=10, font=('Arial', 20, 'bold'), bd=2, relief='solid', command=f6)


excel_window_lbl_time.grid(row = 3, column=0, padx=10, pady=10)
excel_window_ent_exp.grid(row = 3, column=1, padx=10, pady=10)
excel_window_btn_result.grid(row = 4, column=0, padx=10, pady=10)
excel_window_btn_back.grid(row = 5, column=0, padx=10, pady=10)
excel_window.withdraw()

main_window.mainloop()

