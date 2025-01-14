import numpy as np
import cv2
import time


cap = cv2.VideoCapture(0)
# intrinsics = np.mat([[1.42127421e+03, 0.00000000e+00, 9.80978474e+02],
#  [0.00000000e+00, 1.42048319e+03, 5.47188117e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
intrinsics = np.mat([[1.42127421e+03, 0.00000000e+00, 320],
 [0.00000000e+00, 1.42048319e+03, 240],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])



extrinsics = np.mat([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 10],
    [0, 0, 0, 1],
])

def pixel_to_position(pixels):
    # print("pixels shape:", pixels.shape)
    # Camera intrinsic matrix
    K = np.mat([[1.42127421e+03, 0.00000000e+00, 320],
 [0.00000000e+00, 1.42048319e+03, 240],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    u = pixels[0,0]
    v = pixels[0,1]
    # print("x: ", u, " y: ", v)
    d = 1
    c_x, c_y = K[0,2], K[1,2]
    f_x,f_y = K[0,0], K[1,1]
    x = (u - c_x) * d / f_x
    y = (v - c_y) * d / f_y
    z = d

    camera_coords = np.mat([x, y, z, 1]).T
    return camera_coords


  
    


while(True):
   # Capture frame-by-frame
   ret, captured_frame = cap.read()
   
   output_frame = captured_frame.copy()
#    print("image shape: ", output_frame.shape)


   # Convert original image to BGR, since Lab is only available from BGR
   captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
   # First blur to reduce noise prior to color space conversion
   captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)
   # Convert to Lab color space, we only need to check one channel (a-channel) for red here
   captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
   # Threshold the Lab image, keep only the red pixels
   # Possible yellow threshold: [20, 110, 170][255, 140, 215]
   # Possible blue threshold: [20, 115, 70][255, 145, 120]
   captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
   # Second blur to reduce more noise, easier circle detection
   captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
   # Use the Hough transform to detect circles in the image
   circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)
   #time.sleep(0.1)

   # If we have extracted a circle, draw an outline
   # We only need to detect one circle here, since there will only be one reference object
   if circles is not None:
       circles = np.round(circles[0, :]).astype("int")
       cv2.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
       print(pixel_to_position(circles))

    
        


   # Display the resulting frame, quit with q
   cv2.imshow('frame', output_frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

