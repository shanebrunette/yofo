import numpy as np
import cv2
import time

LOCATION = "sample.avi"

def get_video(location):
	# cap = cv2.VideoCapture(location)
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("check file location")
		return 0, None
	else:
		return cap


def show_video(cap):
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	ret, frame = cap.read()
	while ret:
		cv2.imshow('frame', frame)
		if cv2.waitKey(fps) & 0xFF == ord('q'):
			break 
		ret, frame = cap.read()
	cap.release()
	cv2.destroyAllWindows()

def main():
	camera = cv2.VideoCapture(0)

	cv2.namedWindow("tracking")
	ok, image = camera.read()
	if not ok:
		print('Failed to read video')
		exit()
	bbox = cv2.selectROI("tracking", image)
	bbox2 = cv2.selectROI("tracking", image)
	# tracker = cv2.TrackerKCF_create()
	# tracker2 = cv2.TrackerKCF_create()
	tracker = cv2.TrackerGOTURN_create()
	tracker2 = cv2.TrackerGOTURN_create()
	# tracker = cv2.TrackerTLD_create()
	# tracker2 = cv2.TrackerTLD_create()
	multi = cv2.MultiTracker()

	ok, image=camera.read()
	ok = multi.add(tracker, image, bbox)
	ok = multi.add(tracker2, image, bbox2)


	while camera.isOpened():
		ok, image=camera.read()
		if not ok:
			break

		ok, newbox = multi.update(image)


		if ok:
			p1 = (int(newbox[0][0]), int(newbox[0][1]))
			p2 = (int(newbox[0][0] + newbox[0][2]), int(newbox[0][1] + newbox[0][3]))
			cv2.rectangle(image, p1, p2, (200,0,0))

			p1 = (int(newbox[1][0]), int(newbox[1][1]))
			p2 = (int(newbox[1][0] + newbox[1][2]), int(newbox[1][1] + newbox[1][3]))
			cv2.rectangle(image, p1, p2, (200,0,0))

		cv2.imshow("tracking", image)
		k = cv2.waitKey(26) & 0xff
		if k == 27 : break # esc pressed
	


if __name__ == "__main__":
	main()