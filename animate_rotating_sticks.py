# rotate_line_1.py
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from Tkinter import *
import math
root = Tk()
root.title("Rotating line")
cw = 960 # canvas width
ch = 720 # canvas height
chart_1 = Canvas(root, width=cw, height=ch, background="white")
chart_1.grid(row=0, column=0)
cycle_period = 50 # pause duration (milliseconds).

p1_x = 0.0 # the pivot point
p1_y = 0.0 # the pivot point,
p2_x = 30 # the specific point to be rotated
p2_y = 30.0 # the specific point to be rotated.

a_radian = math.atan((p2_y - p1_y)/(p2_x - p1_x))
a_length = math.sqrt((p2_y - p1_y)*(p2_y - p1_y) +\
                                 (p2_x - p1_x)*(p2_x - p1_x))

p1_xb = 0 # the pivot point
p1_yb = 0 # the pivot point,
p2_xb = 500 # the specific point to be rotated
p2_yb = 50 # the specific point to be rotated.

a_radianb = math.atan((p2_yb - p1_yb)/(p2_xb - p1_xb))
a_lengthb = math.sqrt((p2_yb - p1_yb)*(p2_yb - p1_yb) +\
                                 (p2_xb - p1_xb)*(p2_xb - p1_xb))


#big circle
p1_xc = 90.0+50 # the pivot point
p1_yc = 90.0+50 # the pivot point,
p2_xc = 180.0+50 # the specific point to be rotated
p2_yc = 160.0+50 # the specific point to be rotated.

a_radianc = math.atan((p2_yc - p1_yc)/(p2_xc - p1_xc))
a_lengthc = math.sqrt((p2_yc - p1_yc)*(p2_yc - p1_yc) +\
                                 (p2_xc - p1_xc)*(p2_xc - p1_xc))

p1_xc2 = p1_xc
p1_yc2 = p1_yc
p2_xc2 = p2_xc
p2_yc2 = p2_yc

a_radianc2 = math.atan((p2_yc - p1_yc)/(p2_xc - p1_xc))
a_lengthc2 = math.sqrt((p2_yc - p1_yc)*(p2_yc - p1_yc) +\
                                 (p2_xc - p1_xc)*(p2_xc - p1_xc))

a_radianc=3.14
a_radianc2=0.0
a_radian2=0.0
a_radian=0.0

for i in range(1,10000): # end the program after 300 position shifts
    a_radian +=0.00+(800-(p1_y+p1_yc))/10000.0 # incremental rotation of 0.05 radians
    a_radian2 +=0.00+(800-(p1_y+p1_yc2))/10000.0 # incremental rotation of 0.05 radians
    #a_radianb +=0.05 # incremental rotation of 0.05 radians
    a_radianc +=0.05 # incremental rotation of 0.05 radians
    a_radianc2 +=0.05 # incremental rotation of 0.05 radians

    p1_xc = p2_xc - a_lengthc * math.cos(a_radianc)
    p1_yc = p2_yc - a_lengthc * math.sin(a_radianc)

    p1_xc2 = p2_xc2 - a_lengthc2 * math.cos(a_radianc2)
    p1_yc2 = p2_yc2 - a_lengthc2 * math.sin(a_radianc2)
    #chart_1.create_line(p1_xc, p1_yc, p2_xc, p2_yc)

    #chart_1.create_line(50,50,-50,-50)
    #chart_1.create_line(50,500,-50,-50)

    p1_x = p2_x - a_length * math.cos(a_radian)
    p1_y = p2_y - a_length * math.sin(a_radian)
    chart_1.create_line(p1_x+p1_xc, p1_y+p1_yc, p2_x+p1_xc, p2_y+p1_yc)

    p1_x = p2_x - a_length * math.cos(a_radian2)
    p1_y = p2_y - a_length * math.sin(a_radian2)
    chart_1.create_line(p1_x+p1_xc2, p1_y+p1_yc2, p2_x+p1_xc2, p2_y+p1_yc2)

    #p1_xb = p2_xb - a_lengthb * math.cos(a_radianb)
    #p1_yb = p2_yb - a_lengthb * math.sin(a_radianb)
    #chart_1.create_line(p1_xb, p1_yb, p2_xb, p2_yb)

    chart_1.update()
    chart_1.after(cycle_period)
    chart_1.delete(ALL)

root.mainloop()
