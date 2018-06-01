from Xlib import display
import time
from matplotlib import pyplot as plt
import numpy as np
from pynput import keyboard
import threading, time
from datetime import datetime
import csv
import math

def on_press(key):
   global pressed_key
#    if pressed_key == 0:
#      try:
#          print('alphanumeric key {0} pressed'.format(
#              key.char))
#      except AttributeError:
#          print('special key {0} pressed'.format(
#              key))



def on_release(key):
    global pressed_key
    global label
    pressed_key = pressed_key + 1
    print('{0} released   {1}'.format(    key,pressed_key))
    if pressed_key == 1:
       print(">>>>>>>record")
    if pressed_key == 2:
       print("-show")
       # listener.stop()
    if pressed_key == 4:
      try:
        if(key.char ==  'y'):
          print("-Ask to recorded : which label ?")
        else :
          if(key.char == 'n'):
            print("-Ask to not recorded")
            pressed_key = 0
          else:
            print("Accepted caratere y/n")
            pressed_key = pressed_key - 1
      except AttributeError:
         print('Bad caractere')
         pressed_key = pressed_key - 1



    if pressed_key == 5:
      label = key.char
      print("-Corresponding label {0} {1}".format(key,label))
      pressed_key = pressed_key + 1



    # if key == keyboard.Key.esc:
    #     # Stop listener

    #     pressed_key = 0
    #     print(">> stopped :record")

        # return False

# listener = null
def thread1():
    # Collect events until released
    global listener
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release)  as listener:
      listener.join()

def process_data(XTab,YTab):

  XTabCopy = XTab
  YTabCopy = YTab

  # print("XDebut {}".format(XTab))
  if len(XTabCopy) < 17 :
   print("Not enough data")
  else:
    while len(XTabCopy) > 17:

      d_min = 10000;
      p_min  = 0

      p    = p_min;
      i    = p_min;
      i = i + 1

      last = len(XTabCopy) -1

      while ( i < last) :
        d = math.sqrt(math.pow(XTabCopy[p] - XTabCopy[i] , 2) + math.pow(YTabCopy[p] - YTabCopy[i], 2));
        if d < d_min :
          d_min = d;
          p_min = p;

        p = i;
        i = i+1

      p = p_min;
      i = p_min+1;

      ptX = (XTabCopy[p] + XTabCopy[i]) / 2
      ptY = (YTabCopy[p] + YTabCopy[i]) / 2

      XTabCopy[i] = ptX
      YTabCopy[i] = ptY

      XTabCopy.pop(p)
      YTabCopy.pop(p)

  return XTabCopy,YTabCopy

def process_speed(XTab,YTab):
  i = 0
  vx = []
  vy = []
  while i < len(XTab)-1 :
    vx.append(XTab[i+1] - XTab[i])
    vy.append(YTab[i+1] - YTab[i])
    i = i + 1
  return vx,vy

pressed_key = 0
threading.Thread(target = thread1).start()




XTab = []
YTab = []
vx = [0]
vy = [0]

plt.ion()
# start data collection
record = 0

start = datetime.now()
with open('record.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile,    delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # listener.stop()
    spamwriter.writerow(['vx','vy','label'])

    while True:
      if pressed_key == 1:
        d = datetime.now() - start;
        start = datetime.now()

        if len(YTab) >150 :
         del XTab[0]
         del YTab[0]
        data = display.Display().screen().root.query_pointer()._data
        XTab.append(int(data["root_x"]))
        YTab.append(int(1080-data["root_y"]))


        plt.figure(1)
        plt.cla()
        plt.plot(XTab, YTab,'b-')
        plt.axis([0, 1920, 0, 1080])
        plt.pause(0.0005)

      if pressed_key == 2:
        XTab_copy,YTab_copy =process_data(XTab,YTab)
        vx,vy = process_speed(XTab_copy,YTab_copy)
        print('vx = {}'.format((vx)))
        plt.figure(1)
        plt.cla()
        plt.plot(XTab, YTab,'b-')
        plt.plot(XTab_copy, YTab_copy,'r*')
        plt.axis([0, 1920, 0, 1080])

        plt.figure(2)
        plt.cla()
        plt.plot(vx,'r')
        plt.plot(vy,'b')
        plt.show()
        plt.pause(0.0005)

        XTab = []
        YTab = []
        pressed_key = 3
        print('-Keep the data?')


      if pressed_key == 6:
        print("label {}".format(label))
        spamwriter.writerow( vx + vy +[label] )
        time.sleep(0.1)
        pressed_key = 0






