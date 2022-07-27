import sys, os, json, collections, ast
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import tifffile as tf
import time
from scipy.ndimage.measurements import center_of_mass


try:
    from databroker import Broker
    db = Broker.named('hxn')
except:
    print("Not connected to a beamline")
    pass


from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDesktopWidget, QApplication, QSizePolicy


ui_path = os.path.dirname(os.path.abspath(__file__))

cmap_names = ['CET-L13', 'CET-L14', 'CET-L15']
cmap_label1 = ['red', 'green', 'blue']
cmap_dict = {}
for i, name in zip(cmap_names, cmap_label1):
    cmap_dict[name] = pg.colormap.get(i).getLookupTable(alpha=True)


class ptychoWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(ptychoWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'ptychoSave_v2.ui'), self)

        self.config = {"wd": os.getcwd(),
                       "scan_num":'',
                       "detector": "merlin1",
                       "crop_roi": (0,0,0,0),
                       "hot_pixels": [],
                       "outl_pixels" : [],
                       "center_coords":(64,64),
                      "switchXY":False, }

        self.updateDisplayParams(self.config)


        #connections
        self.pb_wd.clicked.connect(self.chooseWD)
        self.pb_load_data.clicked.connect(self.loadData)
        self.le_hot_pixels.editingFinished.connect(self.userUpdateHotPixelList)
        self.le_outl_pixels.editingFinished.connect(self.userUpdateOutlPixelList)
        self.pb_show_corr_image.clicked.connect(lambda:self.showCleanImage(self.images[0], self.config))
        self.pb_add_roi.clicked.connect(self.addROI)
        self.le_scan_num.editingFinished.connect(self.userUpdateHotPixelList)
        self.pb_saveh5.clicked.connect(lambda:self.saveh5(1,1))

        [sb.valueChanged.connect(self.updateROI) for sb in
         [self.sb_roi_xpos,self.sb_roi_ypos,self.sb_roi_xsize,self.sb_roi_ysize]]


        self.show()

    def updateDisplayParams(self, param_dict):

        self.le_wd.setText(param_dict["wd"])
        self.le_scan_num.setText(str(param_dict["scan_num"]))
        index = self.cb_detector.findText(param_dict["detector"], QtCore.Qt.MatchFixedString )
        if index >= 0:
            self.cb_detector.setCurrentIndex(index)

        self.sb_roi_xpos.setValue(param_dict["crop_roi"][0])
        self.sb_roi_ypos.setValue(param_dict["crop_roi"][1])
        self.sb_roi_xsize.setValue(param_dict["crop_roi"][2])
        self.sb_roi_ysize.setValue(param_dict["crop_roi"][3])

        self.input_centerx.setValue(param_dict["center_coords"][0])
        self.input_centery.setValue(param_dict["center_coords"][1])

        self.le_hot_pixels.setText(str(param_dict["hot_pixels"])[1:-1])
        self.le_outl_pixels.setText(str(param_dict["outl_pixels"])[1:-1])

        self.ch_b_switch_xy.setChecked(param_dict["switchXY"])


    def updateSaveConfigFile(self):
        pass
        '''
        self.config = {"wd": self.le_wd.text(),
                       "scan_num": self.le_scan_num.text(),
                       "detector": self.cb_detector.currentText(),
                       "crop_roi": (),
                       "hot_pixels": [],
                       "outl_pixels": [],
                       "center_coords": (64, 64),
                       "switchXY": False, }
        '''

    def chooseWD(self):

        foldername = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.config["wd"] = (str(foldername))
        self.updateDisplayParams(self.config)

    def initializeParameters(self, scan_num):

        sid = int(scan_num)  # testwith 173961

        h = db[sid]

        motors = h.start['motors']
        if motors[0].endswith('y'):
            flipScan = True
        else:
            flipScan = False

        merlins = []

        for name in h.start['detectors']:
            if name.startswith('merlin'):
                merlins.append(name)
        '''
        if len(merlins) > 1:
            det_name = "merlin2"
        else:
            det_name = "merlin1"
        '''
        #todo edit items based on what's in the det list
        det_name = self.cb_detector.currentText()

        if det_name in merlins:

            raw_images = np.squeeze(list(h.data(det_name)))
            ic = np.asfarray(h.table(fill=False)['sclr1_ch4'])
            if ic[0] == 0:
                ic[0] = ic[1]

            ic_ = ic[0] / ic
            ic_norm = np.ones_like(raw_images) * ic_[:, np.newaxis, np.newaxis]

            norm_images = raw_images * ic_norm
            images = np.fliplr(norm_images).transpose(0, 2, 1)
            print(np.shape(images))

            return h, images, det_name, flipScan

        else:
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Error")
            error_message.showMessage(str(f"No {det_name} found"))
            return



    def loadData(self):

        # lists to store pixels values when clicked ; later add to config
        # directly adding to config and displaying could be time consuming
        self.list_of_hot_pixels = []
        self.list_of_outl_pixels = []
        '''
        self.images = tf.imread("merlin_img_stk_10.tiff")
        self.loadAnImage(self.images[0])
        '''

        self.config = {"wd": os.getcwd(),
                       "scan_num":'',
                       "detector": "merlin1",
                       "crop_roi": (0,0,0,0),
                       "hot_pixels": [],
                       "outl_pixels" : [],
                       "center_coords":(64,64),
                      "switchXY":False, }

        try:

            self.config["scan_num"] = int(self.le_scan_num.text())

            #need to be threaded with messages
            self.h, self.images, det_name, flipScan = self.initializeParameters(self.config["scan_num"])

            self.config["switchXY"] = flipScan
            self.config["detector"] = det_name
            self.config["center_coords"] = center_of_mass(self.images[0])[::-1]
            self.updateDisplayParams(self.config)
            self.loadAnImage(self.images[0])
            self.addROI()

        except Exception as e:
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Error")
            error_message.showMessage(str(e))
            pass


    def loadAnImage(self, img):

        try:
            self.image_view.clear()
        except:
            pass
        self.ptychoImage = img
        self.statusbar.showMessage(f"Image Shape = {np.shape(self.ptychoImage)}")
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.image_view.addPlot(title="")
        self.p1.setAspectLocked(True)
        self.p1.getViewBox().invertY(True)
        # Item for displaying image data
        self.img = pg.ImageItem(axisOrder = 'row-major')
        self.p1.addItem(self.img)

        # Create color bar and have it control image levels
        cmap = pg.colormap.getFromMatplotlib('viridis')
        #cmap = pg.colormap.get('bipolar')
        cbi = pg.ColorBarItem(colorMap=cmap)
        cbi.setImageItem(self.img, insert_in=self.p1)
        cbi.setLevels([self.ptychoImage.min(), self.ptychoImage.max()])  # colormap range
        '''
        colormap = cmap_dict['red']
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(colormap)), color=colormap)
        # image = np.squeeze(tf.imread(image_path))
        # set image to the image item with cmap
        '''
        self.img.setImage(self.ptychoImage, opacity=1, lut=cmap.getLookupTable())
        self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        # self.img.translate(100, 50)
        # self.img.scale(0.5, 0.5)

        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent

    def addROI(self):

        try:
            self.p1.removeItem(self.rectROI)
        except:
            pass

        yshape,xshape = np.shape(self.images[0])
        ycen, xcen = center_of_mass(self.images[0])

        self.rectROI = pg.RectROI([int(xcen)-64, int(ycen)-64],
                                  [128, 128],pen='r')
        self.p1.addItem(self.rectROI)
        self.updateROIParams()
        self.rectROI.sigRegionChanged.connect(self.updateROIParams)

    def updateROIParams(self):
        roi_pos = self.rectROI.pos()
        roi_size = self.rectROI.size()
        self.sb_roi_xpos.setValue(int(roi_pos[0]))
        self.sb_roi_ypos.setValue(int(roi_pos[1]))
        self.sb_roi_xsize.setValue(int(roi_size[0]))
        self.sb_roi_ysize.setValue(int(roi_size[1]))

        self.config["crop_roi"] = (int(roi_pos.x()), int(roi_pos.y()), int(roi_size.x()), int(roi_size.y()))

    def updateROI(self):
        try:
            roi_pos = QtCore.QPointF(self.sb_roi_xpos.value(), self.sb_roi_ypos.value())
            roi_size = QtCore.QPointF(self.sb_roi_xsize.value(),self.sb_roi_ysize.value())
            self.rectROI.setPos(roi_pos)
            self.rectROI.setSize(roi_size)
        except RuntimeError:
            pass

        self.config["crop_roi"] = (int(roi_pos.x()), int(roi_pos.y()), int(roi_size.x()), int(roi_size.y()))

    def cropToROI(self, img_stk,dims:tuple):

        xpos,ypos,xsize,ysize = dims

        if dims != (0,0,0,0):
            crop_img_stk = img_stk[:, ypos:ypos+ysize, xpos:xpos+xsize]

        return crop_img_stk


    def updateHotOutlPixels(self):
        #update the pixel values to config
        self.config["hot_pixels"] = self.list_of_hot_pixels
        self.config["outl_pixels"] = self.list_of_outl_pixels

    def userUpdateHotPixelList(self):

        #get the new list of vals from the line edit.
        # Assuming user does not mess with the structure
        #may be change to a list widget later
        newList_str = '[' + self.le_hot_pixels.text() +']'

        # Converting string to list
        self.list_of_hot_pixels = ast.literal_eval(newList_str)
        self.config["hot_pixels"] = self.list_of_hot_pixels

    def userUpdateOutlPixelList(self):

        #get the new list of vals from the line edit.
        # Assuming user does not mess with the structure
        #may be change to a list widget later
        newList_str = '[' + self.le_outl_pixels.text() +']'

        # Converting string to list
        self.list_of_outl_pixels = ast.literal_eval(newList_str)
        self.config["outl_pixels"] = self.list_of_outl_pixels

    def replacePixelValues(self, image, list_pixels, setToZero = False):

        if list_pixels:

            #replace the pixel with neighbor average
            for pixel in list_pixels:


                if setToZero:
                    image[pixel[1], pixel[0]] = 0
                    print(f"{pixel[1], pixel[0]} = 0")

                else:
                    replaceWith = np.mean([image[pixel[0]-1], image[pixel[0]+1],
                                                         image[pixel[1]-1],image[pixel[1]+1]])
                    image[pixel[1],pixel[0]] = int(replaceWith)

                    print(f"{pixel[1], pixel[0]} = {int(replaceWith)}")

            return image

        else:
            print("No Pixel correction")
            print(list_pixels)
            pass

    def showCleanImage(self,img, config_):
        self.updateHotOutlPixels()
        self.replacePixelValues(img, config_["outl_pixels"], setToZero=True)
        self.replacePixelValues(img, config_["hot_pixels"], setToZero=False)
        self.loadAnImage(img)

    def cleanStack(self,img_stk, config_):

        self.updateHotOutlPixels()
        self.mod_img_stk = np.zeros_like(img_stk)

        for n in range(img_stk.shape[0]):

            image = img_stk[n]

            self.replacePixelValues(image, config_["outl_pixels"], setToZero=True)
            self.replacePixelValues(image, config_["hot_pixels"], setToZero=False)
            self.mod_img_stk[n] = image

        return self.mod_img_stk

    def showProcessedCleanImage(self,img_stk, config_, plotAfter = True):

        self.updateHotOutlPixels()
        self.mod_img_stk = np.zeros_like(img_stk)

        for n in range(img_stk.shape[0]):

            image = img_stk[n]

            self.replacePixelValues(image, config_["outl_pixels"], setToZero=True)
            self.replacePixelValues(image, config_["hot_pixels"], setToZero=False)
            self.mod_img_stk[n] = image

        print("image stack updated")

        cx, cy = int(self.config["center_coords"][0]), int(self.config["center_coords"][1])
        n, nn = int(self.config["crop_roi"][-2]), int(self.config["crop_roi"][-1])
        y1, y2 = cy - nn // 2, cy + nn // 2
        x1, x2 = cx - n // 2, cx + n // 2

        # remove bad pixels
        mod_image = self.cleanStack(self.images, self.config)

        tmptmp = mod_image[:, y1:y2, x1:x2]

        data = np.fft.fftshift(tmptmp, axes = [1,2])

        threshold = 1.
        data = data - threshold
        data[data < 0.] = 0.
        data = np.sqrt(data)

        if plotAfter:

            self.loadAnImage(data[0])

        else:
            return data

    def saveh5(self, mesh_flag, fly_flag, distance=0.5):

        df = self.h.table(fill=False)
        bl = self.h.table('baseline')
        plan_args = self.h.start['plan_args']
        motors = self.h.start['motors']

        try:
            angle = bl.zpsth[1]
        except:
            angle = 0

        dcm_th = bl.dcm_th[1]
        energy_kev = 12.39842 / (2. * 3.1355893 * np.sin(dcm_th * np.pi / 180.))
        # energy_kev = bl.energy[0] #replace?
        lambda_nm = 1.2398 / energy_kev

        if mesh_flag:
            if fly_flag:
                x_range = plan_args['scan_end1'] - plan_args['scan_start1']
                y_range = plan_args['scan_end2'] - plan_args['scan_start2']
                x_num = plan_args['num1']
                y_num = plan_args['num2']
            else:
                x_range = plan_args['args'][2] - plan_args['args'][1]
                y_range = plan_args['args'][6] - plan_args['args'][5]
                x_num = plan_args['args'][3]
                y_num = plan_args['args'][7]
            dr_x = 1. * x_range / x_num
            dr_y = 1. * y_range / y_num
            x_range = x_range - dr_x
            y_range = y_range - dr_y
        else:
            x_range = plan_args['x_range']
            y_range = plan_args['y_range']
            dr_x = plan_args['dr']
            dr_y = 0

        if self.config["switchXY"]:

            y = np.array(df[motors[0]])
            x = np.array(df[motors[1]])

        else:
            x = np.array(df[motors[0]])
            y = np.array(df[motors[1]])

        points = np.vstack([x, y])

        cx, cy = int(self.config["center_coords"][0]), int(self.config["center_coords"][1])
        n, nn = int(self.config["crop_roi"][-2]), int(self.config["crop_roi"][-1])
        y1, y2 = cy - nn // 2, cy + nn // 2
        x1, x2 = cx - n // 2, cx + n // 2

        # remove bad pixels
        mod_image = self.cleanStack(self.images, self.config)

        tmptmp = mod_image[:, y1:y2, x1:x2]

        data = np.fft.fftshift(tmptmp, axes = [1,2])

        threshold = 1.
        data = data - threshold
        data[data < 0.] = 0.
        data = np.sqrt(data)

        det_pixel_um = 55.
        det_distance_m = distance

        pixel_size = lambda_nm * 1.e-9 * det_distance_m / (n * det_pixel_um * 1e-6)
        depth_of_field = lambda_nm * 1.e-9 / (n / 2 * det_pixel_um * 1.e-6 / det_distance_m) ** 2
        print('pixel num, pixel size, depth of field: ', n, pixel_size, depth_of_field)


        #with h5py.File(self.config["wd"]+'/h5_data/scan_' + np.str(self.config["scan_num"]) + '.h5', 'w') as hf:
        with h5py.File(self.config["wd"] + '/scan_' + str(self.config["scan_num"]) + '.h5', 'w') as hf:
            dset = hf.create_dataset('diffamp', data=data)
            dset = hf.create_dataset('points', data=points)
            dset = hf.create_dataset('x_range', data=x_range)
            dset = hf.create_dataset('y_range', data=y_range)
            dset = hf.create_dataset('dr_x', data=dr_x)
            dset = hf.create_dataset('dr_y', data=dr_y)
            dset = hf.create_dataset('z_m', data=det_distance_m)
            dset = hf.create_dataset('lambda_nm', data=lambda_nm)
            dset = hf.create_dataset('ccd_pixel_um', data=det_pixel_um)
            dset = hf.create_dataset('angle', data=angle)
            # dset = hf.create_dataset('Ni_xrf',data=Ni_xrf)
            # dset = hf.create_dataset('Au_xrf',data=Au_xrf)

        '''
        #symlink
        src = f'{self.config["wd"]}/h5_data/scan_/{self.config["scan_num"]}.h5'
        dest = f'{self.config["wd"]}/{self.config["scan_num"]}.h5'
        os.symlink(src,dest)
        '''

    def MouseClickEvent(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.img.mapToParent(event.pos())
                i, j = pos.x(), pos.y()
                limits = self.img.mapToParent(QtCore.QPointF(self.ptychoImage.shape[0], self.ptychoImage.shape[1]))
                i = int(np.clip(i, 0, limits.y() - 1))
                j = int(np.clip(j, 0, limits.x() - 1))

                if self.rb_choose_hot.isChecked():

                    if not (i,j) in self.list_of_hot_pixels:
                        self.list_of_hot_pixels.append((i,j))  # if not integer (self.xpixel,self.ypixel)
                        self.le_hot_pixels.setText(str(self.list_of_hot_pixels)[1:-1])

                elif self.rb_choose_outl.isChecked():
                    if not (i,j) in self.list_of_outl_pixels:
                        self.list_of_outl_pixels.append((i,j))  # if not integer (self.xpixel,self.ypixel)
                        self.le_outl_pixels.setText(str(self.list_of_outl_pixels)[1:-1])

                elif self.rb_choose_com.isChecked():
                    self.config["center_coords"] = (i,j)
                    self.input_centerx.setValue(i)
                    self.input_centery.setValue(j)


            else: event.ignore()
        else: event.ignore()

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ptychoImage.shape[1] - 1))
        j = int(np.clip(j, 0, self.ptychoImage.shape[0] - 1))
        val = self.ptychoImage[int(j), int(i)]
        self.p1.setTitle(f'pixel: {i, j}, Intensity : {val:.2f}')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ptychoWindow()
    w.show()
    sys.exit(app.exec_())