import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from scipy.ndimage.measurements import center_of_mass

from databroker import Broker
db = Broker.named('hxn')

def rm_pixel(data,ix,iy):
    data[ix,iy] = np.median(data[ix-1:ix+1,iy-1:iy+1])
    return data

def rm_outlier_pixels(data,hotpixels):
    n = np.size(hotpixels)//2
    for i in range(n):
        x,y = hotpixels[:,i]
        data[x,y] = np.median(data[x-1:x+1,y-1:y+1])
    return data

def find_outlier_pixels(data,tolerance=3,worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset.
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = 10*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    if worry_about_edges == True:
        height,width = np.shape(data)
        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med

    return hot_pixels,fixed_image


def initializeParameters(scan_num):
    sid = np.int(scan_num)

    h = db[sid]
    df = h.table(fill=False)
    bl = h.table('baseline')
    plan_args = h.start['plan_args']
    motors = h.start['motors']
    if motors[0].endswith('y'):
        flipScan = True
    else:
        flipScan = False

    merlins = []

    for name in h.fields():
        if name.startswith('merlin'):
            merlins.append(name)

    if len(merlins) > 1:
        det_name = "merlin2"
    else:
        det_name = "merlin1"

    images = np.squeeze(list(h.data(det_name)))
    print(np.shape(images))

    return det_name,flipScan


def load_data(scan_num,det_name,n,nn,mesh_flag,fly_flag,check_flag):
    sid = np.int(scan_num)

    h = db[sid]
    df = h.table(fill = False)
    bl = h.table('baseline')
    plan_args = h.start['plan_args']

    if "merlin1" in h.fields():
        det_name = "merlin1"

    elif "merlin2" in h.fields():
        det_name = "merlin2"

    images = np.squeeze(list(h.data(det_name)))
    print(np.shape(images))

    try:
        angle = bl.zpsth[1]
    except:
        angle = 0

    dcm_th = bl.dcm_th[1]
    energy_kev = 12.39842 / (2.*3.1355893 * np.sin(dcm_th * np.pi / 180.))
    #energy_kev = bl.energy[0] #replace?

    num_frame, count = np.shape(df)
    #num_frame = 500
    data = []

    if mesh_flag:
        if fly_flag:
            x_range = plan_args['scan_end1']-plan_args['scan_start1']
            y_range = plan_args['scan_end2']-plan_args['scan_start2']
            x_num = plan_args['num1']
            y_num = plan_args['num2']
        else:
            x_range = plan_args['args'][2]-plan_args['args'][1]
            y_range = plan_args['args'][6]-plan_args['args'][5]
            x_num = plan_args['args'][3]
            y_num = plan_args['args'][7]
        dr_x = 1.*x_range/x_num
        dr_y = 1.*y_range/y_num
        x_range = x_range - dr_x
        y_range = y_range - dr_y
    else:
        x_range = plan_args['x_range']
        y_range = plan_args['y_range']
        dr_x = plan_args['dr']
        dr_y = 0
    '''
    if np.abs(angle) <= 45.1:
        x = np.array(df['dssx'])
    else:
        x = np.array(df['dssz'])
    y = np.array(df['dssy'])
    '''
    motors = db[sid].start['motors']
    print(motors)
    x = np.array(df[motors[0]])
    y = np.array(df[motors[1]])
    
    points = np.zeros((2,num_frame))
    points[0,:] = x#[:500]
    points[1,:] = y#[:500]

    ic = np.asfarray(df['sclr1_ch4'])
    if ic[0] == 0:
        ic[0] = ic[1]
    
    '''
    Ni_xrf = np.asfarray(df['Det1_Ni']+df['Det2_Ni']+df['Det3_Ni'])
    Au_xrf = np.asfarray(df['Det1_Au_M']+df['Det2_Au_M']+df['Det3_Au_M'])
    Ni_xrf = Ni_xrf * ic[0] / ic
    #Au_xrf = Ni_xrf
    Au_xrf = Au_xrf * ic[0] / ic
    '''
    for i in range(num_frame):
        #tt = np.flipud(images.get_frame(i)[0]).T
        #tt = np.flipud(np.flipud(images[i,:,:]).T)
        tt = (np.flipud(images[i,:,:]).T)
        nx,ny = np.shape(tt)

        #tt[97:100,114:117] = 0.
        #tt[55:200,200:220] = 0.

        t = tt
        if check_flag:
            t = tt
            plt.figure()
            plt.subplot(221)
            plt.imshow(np.sqrt(t))
            plt.subplot(223)
            plt.imshow(np.log10(t+0.001))
            plt.show()
            print(ic[0])
            #break

        t = t * ic[0] / ic[i]
        if i == 0:
            cx = 98#55
            cy = 115#77
            nx,ny = np.shape(t)
            data = np.zeros((num_frame,n,nn))
            #print(ic[0])
            #hp,tmp = find_outlier_pixels(t)

        #t = rm_outlier_pixels(t,hp)

        #t[77:79,86:89] = 0
        t = rm_pixel(t,94,90)
        t = rm_pixel(t,128,119)
        #t = rm_pixel(t,90,74)
        #t = rm_pixel(t,182,419)
        #t = rm_pixel(t,317,165)
        #t = rm_pixel(t,457,100)
        #t = rm_pixel(t,500,296)

        if check_flag:
            plt.subplot(222)
            plt.imshow(np.sqrt(t))
            plt.subplot(224)
            plt.imshow(np.log10(t+0.001))
            plt.show()

            break
        
        tmptmp = t[cx-n//2:cx+n//2,cy-nn//2:cy+nn//2]
        #print(np.shape(tmptmp))
        #tmptmp = np.zeros((n,nn))
        #tmptmp[36:,:-29] = t[:152,52:]

        data[i,:,:] = np.fft.fftshift(tmptmp)

    threshold = 1.
    data = data - threshold
    data[data < 0.] = 0.
    data = np.sqrt(data)
    return data, angle,x_range, y_range, dr_x, dr_y, points, energy_kev#, Ni_xrf, Au_xrf


def save_data(scan_num,mesh_flag,fly_flag,n,nn,distance,check_flag=0):
    scan_num = np.int(scan_num)
    mesh_flag = np.int(mesh_flag)
    fly_flag = np.int(fly_flag)
    det_name = 'merlin1'
    #energy_kev = 8.37
    det_pixel_um = 55.
    det_distance_m = distance
    n = np.int(n)
    nn = np.int(nn)
    #angle = 10

    data, angle,x_range, y_range, dr_x, dr_y, points, energy_kev = load_data(scan_num,det_name,n,nn,mesh_flag,fly_flag,check_flag)
    print(np.shape(data),'angle: ',angle)
    print('energy:',energy_kev)
    lambda_nm = 1.2398/energy_kev
    pixel_size = lambda_nm * 1.e-9 * det_distance_m / (n * det_pixel_um * 1e-6)
    depth_of_field = lambda_nm * 1.e-9 / (n/2 * det_pixel_um*1.e-6 / det_distance_m)**2
    print('pixel num, pixel size, depth of field: ',n,pixel_size,depth_of_field)

    with h5py.File('./h5_data/scan_'+np.str(scan_num)+'.h5', 'w') as hf:
        dset = hf.create_dataset('diffamp',data=data)
        dset = hf.create_dataset('points',data=points)
        dset = hf.create_dataset('x_range',data=x_range)
        dset = hf.create_dataset('y_range',data=y_range)
        dset = hf.create_dataset('dr_x',data=dr_x)
        dset = hf.create_dataset('dr_y',data=dr_y)
        dset = hf.create_dataset('z_m',data=det_distance_m)
        dset = hf.create_dataset('lambda_nm',data=lambda_nm)
        dset = hf.create_dataset('ccd_pixel_um',data=det_pixel_um)
        dset = hf.create_dataset('angle',data=angle)
        #dset = hf.create_dataset('Ni_xrf',data=Ni_xrf)
        #dset = hf.create_dataset('Au_xrf',data=Au_xrf)

    os.system('ln -s /GPFS/XF03ID1/users/2022Q2/Huang_2022Q2/FCGNM_ptycho/h5_data/scan_'+np.str(scan_num)+'.h5 /GPFS/XF03ID1/users/2022Q2/Huang_2022Q2/FCGNM_ptycho/scan_'+np.str(scan_num)+'.h5')
