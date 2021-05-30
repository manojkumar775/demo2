import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense
from keras.models import model_from_json,load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(934, 765)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(60, 460, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(110, 90, 631, 321))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 30, 531, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(300, 630, 191, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 460, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(60, 550, 161, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(300, 480, 211, 51))
        self.textEdit.setObjectName("textEdit")
        self.Classify_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Classify_2.setGeometry(QtCore.QRect(300, 550, 191, 51))
        self.Classify_2.setObjectName("Classify_2")
        self.Classify_3 = QtWidgets.QPushButton(self.centralwidget)
        self.Classify_3.setGeometry(QtCore.QRect(570, 550, 151, 51))
        self.Classify_3.setObjectName("Classify_3")
        self.Classify_4 = QtWidgets.QPushButton(self.centralwidget)
        self.Classify_4.setGeometry(QtCore.QRect(570, 480, 151, 51))
        self.Classify_4.setObjectName("Classify_4")
        self.Classify_5 = QtWidgets.QPushButton(self.centralwidget)
        self.Classify_5.setGeometry(QtCore.QRect(60, 630, 191, 51))
        self.Classify_5.setObjectName("Classify_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.trainingFunction)
        
        self.Classify_2.clicked.connect(self.pretrainedFunction)

        self.Classify_3.clicked.connect(self.opencvFunction)

        self.Classify_4.clicked.connect(self.datasetFunction)

        self.Classify_5.clicked.connect(self.pretrainedclassifyFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "HAND GESTURE RECOGNITION"))
        self.Classify.setText(_translate("MainWindow", "Classify using CNN"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))
        self.Classify_2.setText(_translate("MainWindow", "Using Pre-Trained Model"))
        self.Classify_3.setText(_translate("MainWindow", "Using Open CV"))
        self.Classify_4.setText(_translate("MainWindow", "Dataset Creation"))
        self.Classify_5.setText(_translate("MainWindow", "Classify Using Pretrained Model"))
    

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") 
        if fileName: 
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) 
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) 
            self.imageLbl.setPixmap(pixmap) 
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) 

    def classifyFunction(self):
        json_file = open('cnnmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("cnnmodel.h5")
        print("Loaded model from disk");
        label=["NONE","ONE","TWO","THREE","FOUR","FIVE"]
        path2=self.file
        print(path2)
        test_image = image.load_img(path2, target_size = (256, 256),grayscale=True)        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        fresult=np.max(result)
        label2=label[result.argmax()]
        print(label2)
        self.textEdit.setText(label2)

    def pretrainedclassifyFunction(self):
        json_file = open('pretrainedmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("pretrainedmodel.h5")
        print("Loaded model from disk");
        label=["NONE","ONE","TWO","THREE","FOUR","FIVE"]
        path2=self.file
        print(path2)
        test_image = image.load_img(path2, target_size = (256, 256),grayscale=True)        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        fresult=np.max(result)
        label2=label[result.argmax()]
        print(label2)
        self.textEdit.setText(label2)

        
    def pretrainedFunction(self):
        IMAGE_SIZE = [256, 256]

        train_path = 'C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/train'
        valid_path = 'C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/test'

        vgg = VGG16(input_shape=IMAGE_SIZE + [1], weights=None,classifier_activation="softmax",include_top=False)

        for layer in vgg.layers:
          layer.trainable = False
          
        folders = glob('C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/train/*')
          

        x = Flatten()(vgg.output)

        prediction = Dense(len(folders), activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=prediction)

        model.summary()
        
        model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        )

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           rotation_range = 12.,
                                           width_shift_range = 0.2,
                                           height_shift_range = 0.2,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory('C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/train',
                                                                 target_size = (256, 256),
                                                                 color_mode = 'grayscale',
                                                                 batch_size = 32,
                                                                 classes = ['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                                 class_mode = 'categorical')

        labels = (training_set.class_indices)
        print(labels)
        test_set = test_datagen.flow_from_directory('C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/test',
                                                            target_size = (256, 256),
                                                            color_mode='grayscale',
                                                            batch_size = 32,
                                                            classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                            class_mode='categorical')

        callback_list = [
                    EarlyStopping(monitor='val_loss',patience=20),
                    ModelCheckpoint(filepath="pretrainedmodel.h5",monitor='val_loss',save_best_only=True,verbose=1)]
        labels2 = (test_set.class_indices)
        print(labels2)

        model.fit_generator(training_set,
                            steps_per_epoch = 80,
                            epochs = 200,
                            validation_data = test_set,
                            validation_steps = 15,
                            callbacks=callback_list
                          )
        
        model_json=model.to_json()
        with open("pretrainedmodel.json", "w") as json_file:
          json_file.write(model_json)
          # serialize weights to HDF5
          model.save_weights("pretrainedmodel.h5")
          print("Saved model to disk")
          self.textEdit.setText("Saved model to disk")

          
    def opencvFunction(self):
        import math
        cap = cv2.VideoCapture(0)
        while(True):

            __,img=cap.read()
            cv2.rectangle(img,(400,400),(50,50),(0,255,0),0)
            crop_img = img[50:400, 50:400]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            value = (35, 35)
            blurred = cv2.GaussianBlur(grey, value, 0)
            _, thresh1 = cv2.threshold(blurred, 127, 255,
                                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            cv2.imshow('Thresholded', thresh1)

            contours = cv2.findContours(thresh1.copy(), \
                   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]

            cnt = max(contours, key = lambda x: cv2.contourArea(x))

            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
            hull = cv2.convexHull(cnt)
            drawing = np.zeros(crop_img.shape,np.uint8)
            cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
            cv2.drawContours(drawing,[hull],0,(0,0,255),0)
            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull)
            count_defects = 0
            cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_img,far,1,[0,0,255],-1)
                #dist = cv2.pointPolygonTest(cnt,far,True)
                cv2.line(crop_img,start,end,[0,255,0],2)
                #cv2.circle(crop_img,far,5,[0,0,255],-1)
            if count_defects == 1:
                cv2.putText(img,"GESTURE ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            elif count_defects == 2:
                str = "GESTURE TWO"
                cv2.putText(img, str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2)
            elif count_defects == 3:
                cv2.putText(img,"GESTURE THREE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            elif count_defects == 4:
                cv2.putText(img,"GESTURE FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            else:
                cv2.putText(img,"Hello World!!!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            cv2.imshow('Gesture', img)
            all_img = np.hstack((drawing, crop_img))
            cv2.imshow('Contours', all_img)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def datasetFunction(self):

        if not os.path.exists("data"):
            os.makedirs("data")
            os.makedirs("data/train")
            os.makedirs("data/test")
            os.makedirs("data/train/0")
            os.makedirs("data/train/1")
            os.makedirs("data/train/2")
            os.makedirs("data/train/3")
            os.makedirs("data/train/4")
            os.makedirs("data/train/5")
            os.makedirs("data/test/0")
            os.makedirs("data/test/1")
            os.makedirs("data/test/2")
            os.makedirs("data/test/3")
            os.makedirs("data/test/4")
            os.makedirs("data/test/5")
            
        mode = 'train'
        directory = 'data/'+mode+'/'

        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()

            frame = cv2.flip(frame, 1)

            count = {'zero': len(os.listdir(directory+"/0")),
                     'one': len(os.listdir(directory+"/1")),
                     'two': len(os.listdir(directory+"/2")),
                     'three': len(os.listdir(directory+"/3")),
                     'four': len(os.listdir(directory+"/4")),
                     'five': len(os.listdir(directory+"/5"))}
            
            cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            

            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
          
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
     
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (64, 64)) 
         
            cv2.imshow("Frame", frame)
            
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
            cv2.imshow("ROI", roi)
            
            interrupt = cv2.waitKey(10)
            if interrupt & 0xFF == 27: # esc key
                break
            if interrupt & 0xFF == ord('0'):
                cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
            if interrupt & 0xFF == ord('1'):
                cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
            if interrupt & 0xFF == ord('2'):
                cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
            if interrupt & 0xFF == ord('3'):
                cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
            if interrupt & 0xFF == ord('4'):
                cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
            if interrupt & 0xFF == ord('5'):
                cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
            
        cap.release()
        cv2.destroyAllWindows()

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")
        #basic cnn
        model = Sequential()
        model.add(Conv2D(8, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(16, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(32, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(128, (3, 3), activation = 'relu'))
        model.add(Conv2D(128, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(256, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(units = 200, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units = 6, activation = 'softmax'))

        model.summary()

        
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 12.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range=0.15,
                                   horizontal_flip = True)
        val_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory('C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/train',
                                                         target_size = (256, 256),
                                                         color_mode = 'grayscale',
                                                         batch_size = 32,
                                                         classes = ['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                         class_mode = 'categorical')

        labels = (training_set.class_indices)
        print(labels)
        val_set = val_datagen.flow_from_directory('C:/Users/M.MANOJ KUMAR REDDY/AppData/Local/Programs/Python/Python38/HandGestureDataset/val',
                                                    target_size = (256, 256),
                                                    color_mode='grayscale',
                                                    batch_size = 32,
                                                    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                    class_mode='categorical')


        callback_list = [
            EarlyStopping(monitor='val_loss',patience=50),
            ModelCheckpoint(filepath="cnnmodel.h5",monitor='val_loss',save_best_only=True,verbose=1)]
        
        labels2 = (val_set.class_indices)
        print(labels2)
        
        model.fit_generator(training_set,
                                 steps_per_epoch = 80,
                                 epochs = 200,
                                 validation_data = val_set,
                                 validation_steps = 15,
                                 callbacks=callback_list
                            )

        model_json=model.to_json()
        with open("cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights("cnnmodel.h5")
            print("Saved model to disk")
            self.textEdit.setText("Saved model to disk")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
