# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import numpy as np
import cv2
from numpy import int16, uint8, log2
import pydvs.generate_spikes as gs
import cython
from testingcode.myfilters import singlenoisefilter

# %%
# grab / rescale frame  
def grab_first(dev, res):
  _, raw = dev.read()
  height, width, _ = raw.shape
  new_height = res
  new_width = int( float(new_height*width)/float(height) )
  col_from = (new_width - res)//2
  col_to   = col_from + res
  img = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(int16),
               (new_width, new_height))[:, col_from:col_to]

  return img, new_width, new_height, col_from, col_to

def grab_frame(dev, width, height, col_from, col_to):
  _, raw = dev.read()
  img = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(int16),
               (width, height))[:, col_from:col_to]

  return img


# %%
#----------------------------------------------------------------------#
#mode 16 32 64 128
#0 up polar 1 down 2 merge pollar

video_dev_id = 1
mode = "128"
cam_res = int(mode)
width = cam_res # square output
height = cam_res
shape = (height, width)
#cam_res = 256 # <- can be done, but spynnaker doesn't suppor such resolution

data_shift = uint8( log2(cam_res) )
up_down_shift = uint8(2*data_shift)
data_mask = uint8(cam_res - 1)

polarity = 2

history_weight = 1.0#1.0
threshold = 12 # ~ 0.05*255
max_threshold = 180 # 12*15 ~ 0.7*255

scale_width = 0
scale_height = 0
col_from = 0
col_to = 0

curr     = np.zeros(shape,     dtype=int16) 
ref      = 128*np.ones(shape,  dtype=int16) 
spikes   = np.zeros(shape,     dtype=int16) 
diff     = np.zeros(shape,     dtype=int16) 
abs_diff = np.zeros(shape,     dtype=int16) 

# just to see things in a window
spk_img  = np.zeros((height, width, 3), uint8)

num_bits = 6   # how many bits are used to represent exceeded thresholds
num_active_bits = 2 # how many of bits are active
log2_table = gs.generate_log2_table(num_active_bits, num_bits)[num_active_bits - 1]
spike_lists = None
pos_spks = None
neg_spks = None
max_diff = 0


# %%
# -------------------------------------------------------------------- #
# inhibition related                                                   #
inh_width = 2
is_inh_on = True
inh_coords = gs.generate_inh_coords(width, height, inh_width)
# -------------------------------------------------------------------- #
# camera/frequency related                                             #

video_dev = cv2.VideoCapture(video_dev_id) # webcam
#video_dev = cv2.VideoCapture('/path/to/video/file') # webcam

print(video_dev.isOpened())

#ps3 eyetoy can do 125fps
try:
  video_dev.set(cv2.CAP_PROP_FPS, 125)
except:
  pass
  
fps = video_dev.get(cv2.CAP_PROP_FPS)
if fps == 0.0:
  fps = 125.0
max_time_ms = int(1000./float(fps))


# %%

def Base_work(curr,spikes):
    # get an image from video source
  curr[:] = grab_frame(video_dev, scale_width,  scale_height, col_from, col_to)
  #curr = cv2.medianBlur(curr,5)
  imgraw=np.zeros((128,128,3),np.uint8)
  imgraw[:,:,0]=curr
  imgraw[:,:,1]=curr
  imgraw[:,:,2]=curr
 
  # do the difference
  diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference(curr, ref, threshold)
  
  # inhibition ( optional ) 
  if is_inh_on:
    spikes[:] = gs.local_inhibition(spikes, abs_diff, inh_coords, 
                                 width, height, inh_width)
  
  # update the reference
  ref[:] = gs.update_reference_time_binary_thresh(abs_diff, spikes, ref,
                                               threshold, max_time_ms,
                                               num_active_bits,
                                               history_weight,
                                               log2_table)
  
  # convert into a set of packages to send out
  neg_spks, pos_spks, max_diff = gs.split_spikes(spikes, abs_diff, polarity)
  
  # this takes too long, could be parallelized at expense of memory
  spike_lists = gs.make_spike_lists_time_bin_thr(pos_spks, neg_spks,
                                              max_diff,
                                              up_down_shift, data_shift, data_mask,
                                              max_time_ms,
                                              threshold, 
                                              max_threshold,
                                              num_bits,
                                              log2_table)
  
  spk_img[:] = gs.render_frame(spikes, curr, cam_res, cam_res, polarity)

  #show
  fm=np.copy(spikes.astype(uint8))
  cv2.imshow (WINDOW_NAME, spk_img.astype(uint8)) 
  cv2.putText(fm,f'FPS: {int(fps)}',(0,0),cv2.FONT_HERSHEY_PLAIN,3,(8,255,8),3)
  return imgraw

# %%
def get_white_frame(spikes):
  rows,cols = np.where(spikes==-1)
  imgesR=np.zeros([width,height,1],np.uint8)
  imgesR[rows,cols]=255
  rows,cols = np.where(spikes==1)
  imgesG=np.zeros([width,height,1],np.uint8)
  imgesG[rows,cols]=255
  rows,cols = np.where(spikes!=0)
  imges=np.zeros([width,height,1],np.uint8)
  imges[rows,cols]=255
  return imgesR,imgesG,imges


# %%
def drawobject(figname,img):


  ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  #img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  imges11=np.zeros([width,height,3],np.uint8)
  cv2.drawContours(imges11,contours,-1,(80,60,0),3)
  #cv2.imshow('output1',imges11)

  try: hierarchy = hierarchy[0]
  except: hierarchy = []

  min_x, min_y = width, height
  max_x = max_y = 0

  ar=[]
      # computes the bounding box for the contour, and draws it on the frame,
  for contour, hier in zip(contours, hierarchy):
      (x,y,w,h) = cv2.boundingRect(contour)
      min_x, max_x = min(x, min_x), max(x+w, max_x)
      min_y, max_y = min(y, min_y), max(y+h, max_y)
      if w > 40 and h > 40  and w<150 and h<150:
          cv2.rectangle(imges11, (x,y), (x+w,y+h), (0, 0, 255), 2)
          ar.append([x,y,x+w,x+h])
  #if max_x - min_x > 10 and max_y - min_y > 10 and max_y - min_y < 150 and max_x - min_x<150:
  #    cv2.rectangle(imges11, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
  cv2.imshow(figname,imges11)
  return ar


# %%
def base_filter(current,imgesR,imgesG,imges):
  rir=imgesR
  rig=imgesG
  rii=imges

  #sow all
  Hori = np.concatenate((imges, imgesR), axis=1)
  Hori = np.concatenate((Hori, imgesG), axis=1)
  cv2.imshow ("Hori   N  R   G", Hori)
  #sow filtered
  fi=np.zeros_like(rii)
  fr=np.zeros_like(rir)
  fg=np.zeros_like(rig)
  singlenoisefilter(fr,rir,128,3)
  singlenoisefilter(fg,rig,128,3)
  singlenoisefilter(fi,rii,128,3)

  Hori2 = np.concatenate((rii, rir), axis=1)
  Hori2 = np.concatenate((Hori2, rig), axis=1)
  cv2.imshow ("filtered   N  R   G", Hori2)

  #--------------------------------
  #adding some thing
  kernel = np.ones((5,5),np.uint8)
  dili = cv2.dilate(rii,kernel,iterations=1)
  dilr = cv2.dilate(rir,kernel,iterations=5)
  dilg = cv2.dilate(rig,kernel,iterations=1)

  Hori3 = np.concatenate((dili, dilr), axis=1)
  Hori3 = np.concatenate((Hori3, dilg), axis=1)
  cv2.imshow ("dil   N  R   G", Hori3)

  #draw count
  ar=drawobject("r countur",dilr)

  cu=current
  for x in ar:
      cv2.rectangle(cu, (x[0],x[1]), (x[2],x[3]), (0, 0, 255), 2)
      cv2.putText(cu,"O",(x[0],x[1]),cv2.FONT_HERSHEY_PLAIN,2,(8,255,8),1)
  
  cv2.imshow ("object", cu)




  imgG = dilg
  imgR = dilr
  img  = dili

  return img,imgG,imgR
# %%
#---------------------- main loop -------------------------------------#
LOOP_START =True
if video_dev.isOpened()==False:
    print("error in camera")
    LOOP_START = False
    

WINDOW_NAME = 'spikes'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

is_first_pass = True
start_time = time.time()
end_time = 0
frame_count = 0

pTime =0
curr[:], scale_width, scale_height, col_from, col_to = grab_first(video_dev, cam_res)
while(LOOP_START):
  #get spikes and prepare them  
  d3img=Base_work(curr,spikes)

  #get spike on white frame
  imgesR,imgesG,imges = get_white_frame(spikes)

  #---------------------------------------------------
  #first filter
  img,imgG,imgR = base_filter(d3img,imgesR.squeeze(),imgesG.squeeze(),imges.squeeze())

 
  #==============================================================
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  end_time = time.time()
  
  if end_time - start_time >= 1.0:
    print("%d frames per second"%(frame_count))
    frame_count = 0
    start_time = time.time()
  else:
    frame_count += 1

cv2.destroyAllWindows()


# %%
