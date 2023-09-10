
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QFileDialog
import sys
import cv2
import numpy as np
import time
import imutils


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 564)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 10, 900, 41))
        self.label.setStyleSheet("font: 81 18pt \"Rockwell Extra Bold\";\n" "color: rgb(170, 0, 0);")
        self.label.setObjectName("label")
        self.image1 = QtWidgets.QLabel(self.centralwidget)
        self.image1.setGeometry(QtCore.QRect(155, 80, 661, 350))
        self.image1.setText("")
        self.image1.setObjectName("image1")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(350, 440, 171, 51))
        self.pushButton.setStyleSheet("font: 75 12pt \"MS Shell Dlg 2\";\n" "color: rgb(170, 0, 0);")
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "DETECTION AND CLASSIFICATION OF SATELLITE IMAGES"))
        self.pushButton.setText(_translate("MainWindow", "UPLOAD IMAGE"))
        self.pushButton.clicked.connect(self.classify)


    def classify(self):
        # Load Yolo
        net = cv2.dnn.readNet("data/car.weights", "data/car.cfg")
        classes = []

        with open("data/car.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        color = (255,0,0)
        filename = QFileDialog.getOpenFileName(None, "BROWSE FILE", "", "")
        path = filename[0]
        # Loading image
        img = cv2.imread(path)
        img = imutils.resize(img,width = 700, height=500)
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        pngfile = QPixmap(path)
        pixmap2 = pngfile.scaledToWidth(550)
        pixmap3 = pngfile.scaledToHeight(300)
        self.image1.setPixmap(pixmap3)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
        #print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y-5), font, 0.9, color, 1)
                
        frame = img
        
        net = cv2.dnn.readNet("data/obj.weights","data/obj.cfg") # Original yolov3
        #net = cv2.dnn.readNet("yolov3-tiny.weights","yolov93-tiny.cfg") #Tiny Yolo
        classes = []
        with open("data/obj.names","r") as f:
            classes = [line.strip() for line in f.readlines()]


        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


        colors= np.random.uniform(0,255,size=(len(classes),3))

        #loading image
        font = cv2.FONT_HERSHEY_PLAIN
        #frame = cv2.imread("D:/Projects/ImageProcessing/cambridge/custom_dataset/MOS80.png")
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #reduce 416 to 320    

        net.setInput(blob)
        outs = net.forward(outputlayers)
        #print(outs[1])


        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        TrackedIDs = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    #onject detected
                    
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)                
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence= confidences[i]
                color = colors[class_ids[i]]          
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+str(confidence),(x,y+10),font,0.9,(0,255,0),1)
        #frame = imutils.resize(frame,width = 500)
        cv2.imshow("Image",frame)
        key = cv2.waitKey(0) #wait 1ms the loop will start again and we will process the next frame    
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
