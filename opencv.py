import cv2, time
print('Press Esc to exit')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
imgWindow = cv2.namedWindow('test', cv2.WINDOW_NORMAL)

def detect_face():
	capInput = cv2.VideoCapture(0)
	# 避免处理时间过长造成画面卡顿
	nextCaptureTime = time.time()
	faceRects = []
	color = (255,0,0)
	if not capInput.isOpened(): print('Capture failed because of camera')
	while 1:
		ret, img = capInput.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if nextCaptureTime < time.time():
			nextCaptureTime = time.time() + 0.1
			faceRects = faceCascade.detectMultiScale(gray, 1.3, 5)
		if len(faceRects)>0:
			for faceRect in faceRects:
				x, y, w, h = faceRect
				cv2.rectangle(img, (x, y), (x+w, y+h), color)
		cv2.imshow("test", img)
		key=cv2.waitKey(10)
		c = chr(key & 255)
		if c in ['q', 'Q', chr(27)]:
			break
	capInput.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	detect_face()
