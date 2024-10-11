import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

input_image_path="01.png"

img=jetson.utils.loadImage(input_image_path)

if img is not None:
	detections=net.Detect(img)

	for detection in detections:
		class_id=detection.ClassID
		confidence=detection.Confidence
		left=detection.Left
		top = detection.Top
		right=detection.Right
		bottom = detection.Bottom

		width = right - left
		height = bottom - top
		area=width*height
		center_x = (left + right) / 2
		center_y = (top + bottom) / 2

		print("--ClassID:{}".format(class_id))
		print("--Confidecnce:{}".format(confidence))
		print("--Left:{}".format(left))
		print("--Right:{}".format(right))
		print("--Width:{}".format(width))
		print("--Height:{}".format(height))
		print("--Area:{}".format(area))
		print("--Center:{}".format((center_x,center_y)))




	
