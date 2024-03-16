import cv2
import numpy as np
from openvino.runtime import Core
from picamera2 import Picamera2

def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image /= 255.0
    return image

core = Core()
model_onnx = core.read_model(model='weights/OV_enhance_color-llie-ResCBAM_g.bin')

device = 'MYRIAD'

# Load model on device
compiled_model = core.compile_model(model=model_onnx, device_name=device)
output_layer = compiled_model.output(0)

# Grab images as numpy arrays and leave everything else to OpenCV.

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (256, 256)}))
FrameRate = 15
frame_time = 1000000 // FrameRate # ns

picam2.start()
picam2.set_controls({"FrameDurationLimits":(frame_time,frame_time)})

import time
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 0.7
fontColor              = (0,0,0)
thickness              = 2
lineType               = 2

size = (256*4, 256)
result_vid = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, size,)

while True:
    ini_time = time.time()
    im = picam2.capture_array()
    image = cv2.cvtColor(im , cv2.COLOR_RGBA2RGB)
    w, h, _ = image.shape
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = normalize(resized_image)
    
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)
    
    result = compiled_model([normalized_input_image])[output_layer]
    result_image = result[0].transpose((1,2,0))*255
    resized_result_image = (cv2.resize(result_image,(w, h))).astype(np.uint8)
    print(resized_result_image.shape)
    
    img_gray = cv2.cvtColor(resized_result_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
    edges = cv2.merge([edges]*3)
    print(edges.shape)    
    resized_image = cv2.resize(resized_image, (w, h))
    img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
    edges1 = cv2.merge([edges1]*3)
    print(edges1.shape)
    
    resized_result_image = np.hstack([resized_result_image, edges,resized_image, edges1])
    img_sz = 1600
    resized_result_image = cv2.resize(resized_result_image, (img_sz, int(img_sz/4/w*h)))

    print(time.time()-ini_time)

    cv2.putText(resized_result_image,'FPS: '+str(1/(time.time()-ini_time)), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    
    cv2.imshow("Camera", resized_result_image)
    result_img_vid = cv2.resize(resized_result_image, size)
    result_vid.write(result_img_vid)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

result_vid.release()
cv2.destroyAllWindows()

