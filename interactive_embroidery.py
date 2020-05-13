#the one and the only
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
from PIL import Image
from matplotlib import pyplot as plt
import keyboard
import math
from itertools import islice
import math
import random
import json



img = cv.imread('/Users/mac/Desktop/thesis_img_processing/opencv0.jpg',0)

#target_img = "'/Users/mac/Desktop/thesis_img_processing/opencv0.jpg'"
#activate webcam and take picture
def capture():
	camera_0 = cv.VideoCapture(0)
	camera_1 = cv.VideoCapture(1)

	for i in range(10): 
	    ret, frame = camera_0.read()
	    cv.imwrite('opencv'+str(i)+'.jpg', frame)
	camera_0.release()
	cv.destroyAllWindows()

def denoise(img):
	img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	return (dst)

def findSkeleton(img):
	#img = cv.imread('/Users/mac/Desktop/thesis_img_processing/opencv0.jpg',0)
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
	
	element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
	done = False

	while( not done):
	    eroded = cv.erode(img,element)
	    temp = cv.dilate(eroded,element)
	    temp = cv.subtract(img,temp)
	    skel = cv.bitwise_or(skel,temp)
	    img = eroded.copy()
	    zeros = size - cv.countNonZero(img)
	    if zeros==size:
	        done = True

	#original points
	pts = []
	for i in range(skel.shape[0]):
		for j in range(skel.shape[1]):
			if skel[i][j] == 255:
				pts.append((i,j))
	print ("original",len(pts))

	f = open("pointsOnCurve", "w")
	for pt in pts:
		f.write(str(pt))
	f.close()

	#reduced points
	size = 2
	pts = []
	temp = []
	for i in range(0,skel.shape[0],2):
		for j in range(0,skel.shape[1],2):
			total_x = 0
			total_y = 0
			count = 0
			for x in range(2):
				for y in range (2):
					if (i+x <skel.shape[0]) and (j+y <skel.shape[1]) and (skel[i+x][j+y] == 255):
						count += 1
						total_x += i+x
						total_y += j+y
			if count == 0:
				continue
			else:
				temp.append((total_x/count, total_y/count))

	print ("reduced",len(temp))
	

	#show the skeleton result
	cv.imshow("skel",skel)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return(temp)

def findContour():
	img = cv.imread('/Users/mac/Desktop/thesis_img_processing/opencv0.jpg',0)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
	_, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
	plt.imshow(binary, cmap="gray")
	plt.show()

	contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	img = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
	plt.imshow(img)
	plt.show()

#ArUco Markers
def generate_markers():
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

	fig = plt.figure()
	nx = 2
	ny = 2
	for i in range(1, nx*ny+1):
	    ax = fig.add_subplot(ny,nx, i)
	    img = aruco.drawMarker(aruco_dict,i, 700)
	    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
	    ax.axis("off")

	plt.savefig("_data/markers.pdf")
	plt.show()

#get the edge of the canvas
def find_edges():

	frame = cv.imread("/Users/mac/Desktop/thesis_img_processing/opencv0.jpg")
	height, width, channels = frame.shape
	#print("img info",height, width, channels)
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	f = open("boundary.txt", "w")

	#get center point of the canvas
	tempx = 0
	tempy = 0
	for corner in corners:
		for pts in corner:
			for i in pts:
				tempx += i[0]
				tempy += i[1]

	centerX = int(tempx)/16
	centerY = int(tempy)/16

	#create dictionary
	pts_dist = dict()
	for corner in corners:
		for pts in corner:
			for i in pts:
				pts_dist[(i[0],i[1])] = math.sqrt(((i[0]) - centerX)**2 + ((i[1]) - centerY)**2)

	
	#get the closest 4 points
	sort_by_dist = sorted( pts_dist.items(), key=lambda x: x[1])
	edges = sort_by_dist[:4]
	#update global variable
	

	for i in edges:
		f.write("(" + str(i[0][0]) + ", " + str(i[0][1]) + ")")
	f.close()		
	
	#show markers and ids
	plt.figure()
	plt.imshow(frame_markers)
	for i in range(len(ids)):
		c = corners[i][0]
		plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
	plt.legend()
	plt.show()

	return edges

#get points within the traget area
def get_target(all_points, edge_point):
	pts = all_points
	#find_bonds
	tempx = []
	tempy = []
	for i in edge_point:
		tempy.append(i[0][0])
		tempx.append(i[0][1])

	tempx = sorted(tempx)
	tempy = sorted(tempy)
	left_bond = tempx[0]
	right_bond = tempx[3]
	lower_bond = tempy[0]
	upper_bond = tempy[3]

	print("boundary:","left_bond",left_bond,"right_bond",right_bond,"lower_bond",lower_bond,"upper_bond",upper_bond)

	target_pts = []
	for i in range(len(pts)):
		#if float(pts[i][0]) > float(left_x) and float(ptx[i]) < float(right_x) and float(pty[i]) > float(lower_y) and float(pty[i]) < float(upper_y) :
		if pts[i][0] > left_bond and pts[i][0] < right_bond and pts[i][1] > lower_bond and pts[i][1] < upper_bond :
			target_pts.append(pts[i])

	return (target_pts)

def calculateDistance(x1,y1,x2,y2):
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
	return dist

# 1. starting from one point, add the point to the final_list
# 2. get its nearest neighbor, repeat step 1, until all points are added to final_list
# []: orignal, visited, sorted_sub
# [[]]: final

def sort_points(target_point):

	pts = target_point
	#print("len of pts",len(pts))
	visited = set()
	final = []
	
	#initial the first point
	cur = random.randint(0, len(pts))
	sorted = [cur]
	visited.add(cur)

	while len(visited) != len(pts):
		#10 is threshold
		tar_dis = 10
		tar_idx = -1
		for j in range(len(pts)):
			if cur != j and (j not in visited):
				if tar_dis >  calculateDistance(pts[cur][0],pts[cur][1],pts[j][0],pts[j][1]):
					tar_idx = j
					
		if tar_idx != -1:
			sorted.append(tar_idx)
			visited.add(tar_idx)
			cur = tar_idx
		else:
			final.append(sorted)

			for next_pt in range(len(pts)):
				if next_pt not in visited:
					cur = next_pt
					break
			sorted = [cur]
			visited.add(cur)

	#at this point, final is a list of index
	print("final list length",len(final))
	#now, a list of coordinates
	for i in range(len(final)) :
		for j in range(len(final[i])):
			index = final[i][j]
			final[i][j] = pts[index]


	#get a set of svg to sketch RNN
	#f_svg = open("pointsToRnn", "w")
	total_num = len(final)
	id_val = random.randint(0, total_num)

	print("id_val=",id_val)
	print("final[0]=",final[0])
	print("final[id_val]=",final[id_val])

	with open('data.json', 'w') as f:
		json.dump(final[id_val], f)



	#random pick a sublist?

	f1 = open("pointsOnCurve_x", "w")
	f2 = open("pointsOnCurve_y", "w")
	for sub in final:
		if len(sub) > 10:
			for i in range(len(sub)):
				f1.write(str(sub[i][0])+ ",")
				f2.write(str(sub[i][1])+ ",")
				if i == (len(sub)-1):
					f1.write(str(sub[i][0]) + "\n")
					f2.write(str(sub[i][1])	+ "\n")
			f1.write("---\n ")
			f2.write("---\n ")
	f1.close()	
	f2.close()	


all_points = []
target_point = []
edge_point = []

#denoise()
capture()
all_points = findSkeleton(img)
edge_point = find_edges()
target_point = get_target(all_points, edge_point)
sort_points(target_point)
#findCurve()
#findContour()
print("done!")
	
