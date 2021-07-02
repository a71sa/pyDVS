import time
import numpy as np
import cv2
from numpy import int16, uint8, log2
import pydvs.generate_spikes as gs


# -------------------------------------------------------------------- #
# grab / rescale frame                                                 #

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

#-------------------------------------------------
#----------------------------------------------------------------------#
#mode 16 32 64 128
#0 up polar 1 down 2 merge pollar

video_dev_id = 0
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

history_weight = 1.0
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
#-----------------------------------------------------

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
#-------------------------------------------------------------------
#--------------------------------------------------------------------
#----------------------------------------------------------------
#main algorithm
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
  # get an image from video source
  curr[:] = grab_frame(video_dev, scale_width,  scale_height, col_from, col_to)
  
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

  #cTime = time.time()
  #fps = 1 / (cTime - pTime)
  #pTime = cTime
    
  #show
  fm=np.copy(spikes.astype(uint8))
  cv2.imshow (WINDOW_NAME, spk_img.astype(uint8)) 
  cv2.putText(fm,f'FPS: {int(fps)}',(0,0),cv2.FONT_HERSHEY_PLAIN,3,(8,255,8),3)

  #---------------------------------------------------
  rows,cols = np.where(spikes==-1)
  imges=np.zeros([128,128,1],np.uint8)
  imges[rows,cols]=255

  kernel = np.ones((3,3),np.uint8)
  #opening = cv2.morphologyEx(fm, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(imges, cv2.MORPH_CLOSE, kernel)
  dilation = cv2.dilate(closing,kernel,iterations = 3)
  opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

  img=opening

  cv2.imshow ("spikes1", img) 

  # Take only region of logo from logo image.
  img2_fg = cv2.bitwise_and(spk_img,spk_img,mask = img) 
  cv2.imshow('output',img2_fg)
  #contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(img,contours,-1,(255,255,255),3)  
  #cv2.imshow('output',img)

  ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  #img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  imges11=np.zeros([128,128,3],np.uint8)
  cv2.drawContours(imges11,contours,-1,(255,0,255),3)
  cv2.imshow('output1',imges11)

  try: hierarchy = hierarchy[0]
  except: hierarchy = []

  min_x, min_y = width, height
  max_x = max_y = 0

  # computes the bounding box for the contour, and draws it on the frame,
  for contour, hier in zip(contours, hierarchy):
      (x,y,w,h) = cv2.boundingRect(contour)
      min_x, max_x = min(x, min_x), max(x+w, max_x)
      min_y, max_y = min(y, min_y), max(y+h, max_y)
      if w > 10 and h > 10  and w<150 and h<150:
          cv2.rectangle(imges11, (x,y), (x+w,y+h), (255, 0, 0), 2)

  if max_x - min_x > 10 and max_y - min_y > 10 and max_y - min_y < 150 and max_x - min_x<150:
      cv2.rectangle(imges11, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
  cv2.imshow('output2',imges11)
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