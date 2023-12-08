from reacher import forward_kinematics
from reacher import inverse_kinematics
from reacher import reacher_robot_utils
from reacher import reacher_sim_utils
import pybullet as p
import time
import contextlib
import numpy as np
from absl import app
from absl import flags
from pupper_hardware_interface import interface
from sys import platform
import cv2

flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("ik"          , False, "Whether to control arms through cartesian coordinates(IK) or joint angles")
flags.DEFINE_list("set_joint_angles", [], "List of joint angles to set at initialization.")
flags.DEFINE_bool("red_dot", False, "Whether to control using red dot detection")
FLAGS = flags.FLAGS

KP = 5.0  # Amps/rad
KD = 2.0  # Amps/(rad/s)
MAX_CURRENT = 3  # Amps

UPDATE_DT = 0.01  # seconds

HIP_OFFSET = 0.0335  # meters
L1 = 0.08  # meters
L2 = 0.11  # meters

last_xyz = None

def pixel_to_position(pixels):
    pixels = pixels.reshape(1, -1)
    # print("pixels shape:", pixels.shape)
    # Camera intrinsic matrix
    K = np.mat([[1.42127421e+03, 0.00000000e+00, 320],
    [0.00000000e+00, 1.42048319e+03, 240],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    u = pixels[0,0]
    v = pixels[0,1]
    d = 1
    c_x, c_y = K[0,2], K[1,2]
    f_x,f_y = K[0,0], K[1,1]
    x = (u - c_x) * d / f_x
    y = (v - c_y) * d / f_y
    z = d
    print("u:", u, "v:", v)
    print("x:", x, "y:", y, "z:", z)
    camera_coords = np.array([x,y,z])
    return camera_coords

def find_dot(captured_frame):
  # Capture frame-by-frame
   
   output_frame = captured_frame.copy()
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
       print("FOUND A CIRCLE")
       circles = np.round(circles[0, :]).astype("int")
      #  cv2.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
      #  cv2.imshow('frame', output_frame)
       print(circles[0])
       return circles[0]
   else:
     print("NO CIRCLE")
     return None


def main(argv):
  run_on_robot = FLAGS.run_on_robot
  if(FLAGS.red_dot):  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  reacher = reacher_sim_utils.load_reacher()
  print("reacher loaded")

  # Sphere markers for the students' FK solutions
  shoulder_sphere_id = reacher_sim_utils.create_debug_sphere([1, 0, 0, 1])
  elbow_sphere_id    = reacher_sim_utils.create_debug_sphere([0, 1, 0, 1])
  foot_sphere_id     = reacher_sim_utils.create_debug_sphere([0, 0, 1, 1])
  target_sphere_id   = reacher_sim_utils.create_debug_sphere([1, 1, 1, 1], radius=0.01)

  joint_ids = reacher_sim_utils.get_joint_ids(reacher)
  param_ids = reacher_sim_utils.get_param_ids(reacher, FLAGS.ik)
  reacher_sim_utils.zero_damping(reacher)

  p.setPhysicsEngineParameter(numSolverIterations=10)

  # Set up physical robot if we're using it, starting with motors disabled
  if run_on_robot:
    serial_port = reacher_robot_utils.get_serial_port()
    hardware_interface = interface.Interface(serial_port)
    time.sleep(0.25)
    hardware_interface.set_joint_space_parameters(kp=KP, kd=KD, max_current=MAX_CURRENT)
    hardware_interface.deactivate()

  # Whether or not the motors are enabled
  motor_enabled = False
  if run_on_robot:
    mode_text_id = p.addUserDebugText(f"Motor Enabled: {motor_enabled}", [0, 0, 0.2])
  def checkEnableMotors():
    nonlocal motor_enabled, mode_text_id

    # If spacebar (key 32) is pressed and released (0b100 mask), then toggle motors on or off
    if p.getKeyboardEvents().get(32, 0) & 0b100:
      motor_enabled = not motor_enabled
      if motor_enabled:
        hardware_interface.set_activations([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
      else:
        hardware_interface.deactivate()
      p.removeUserDebugItem(mode_text_id)
      mode_text_id = p.addUserDebugText(f"Motor Enabled: {motor_enabled}", [0, 0, 0.2])

  # Control Loop Variables
  p.setRealTimeSimulation(1)
  counter = 0
  last_command = time.time()
  joint_angles = np.zeros(3)
  if flags.FLAGS.set_joint_angles:
    # first the joint angles to 0,0,0
    for idx, joint_id in enumerate(joint_ids):
      p.setJointMotorControl2(
        reacher,
        joint_id,
        p.POSITION_CONTROL,
        joint_angles[idx],
        force=2.
      )
    joint_angles = np.array(flags.FLAGS.set_joint_angles, dtype=np.float32)
    # Set the simulated robot to match the joint angles
    for idx, joint_id in enumerate(joint_ids):
      p.setJointMotorControl2(
        reacher,
        joint_id,
        p.POSITION_CONTROL,
        joint_angles[idx],
        force=2.
      )

  print("\nRobot Status:\n")
  xyz = np.array([0,0,0])
  # Main loop
  while (True):
    ret, captured_frame = cap.read()
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    # Whether or not to send commands to the real robot
    enable = False

    # If interfacing with the real robot, handle those communications now
    if run_on_robot:
      hardware_interface.read_incoming_data()
      checkEnableMotors()

    # Determine the direction of data transfer
    real_to_sim = not motor_enabled and run_on_robot
   
    # Control loop
    if time.time() - last_command > UPDATE_DT:
      last_command = time.time()
      counter += 1

      # Read the slider values
      try:
        slider_values = np.array([p.readUserDebugParameter(id) for id in param_ids])
      except:
        pass
      if FLAGS.ik:
        # if not FLAGS.red_dot:
        xyz = slider_values
        p.resetBasePositionAndOrientation(target_sphere_id, posObj=xyz, ornObj=[0, 0, 0, 1])
      else:
        joint_angles = slider_values
        enable = True
      
      
      # #overwrite whatever the sliders say if we are using red_dot
      if FLAGS.red_dot:
        # print("red_Dot enabled")
        circle = find_dot(captured_frame)
        if circle is not None:
          coords = pixel_to_position(circle)
          xyz=coords
          xyz[2] = xyz[1]
          xyz[1] = 0.175
          print("Updating:", xyz)
          cv2.circle(captured_frame, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)
        cv2.imshow("frame", captured_frame)


      # If IK is enabled, update joint angles based off of goal XYZ position
      if FLAGS.ik or FLAGS.red_dot:
          print("SENDING THIS xyz:", xyz)
          #p.resetBasePositionAndOrientation(target_sphere_id, posObj=xyz, ornObj=[0, 0, 0, 1])
          ret = inverse_kinematics.calculate_inverse_kinematics(xyz, joint_angles[:3])
          if ret is not None:
            enable = True
            # Wraps angles between -pi, pi
            joint_angles = np.arctan2(np.sin(ret), np.cos(ret))
            print("<<JOINT ANGLES>>", joint_angles)

            # Double check that the angles are a correct solution before sending anything to the real robot
            pos = forward_kinematics.fk_foot(joint_angles[:3])[:3,3]
            if np.linalg.norm(np.asarray(pos) - xyz) > 0.5:
              joint_angles = np.zeros_like(joint_angles)
              if flags.FLAGS.set_joint_angles:
                joint_angles = np.array(flags.FLAGS.set_joint_angles, dtype=np.float32)
              print("Prevented operation on real robot as inverse kinematics solution was not correct")

      



      # If real-to-sim, update the joint angles based on the actual robot joint angles
      if real_to_sim:
        joint_angles = hardware_interface.robot_state.position[6:9]
        joint_angles[0] *= -1

      # Set the simulated robot to match the joint angles
      for idx, joint_id in enumerate(joint_ids):
        p.setJointMotorControl2(
          reacher,
          joint_id,
          p.POSITION_CONTROL,
          joint_angles[idx],
          force=2.
        )

      # Set the robot angles to match the joint angles
      if run_on_robot and enable:
        full_actions = np.zeros([3, 4])
        full_actions[:, 2] = joint_angles
        full_actions[0, 2] *= -1

        # Prevent set_actuator_positions from printing to the console
        with contextlib.redirect_stdout(None):
          hardware_interface.set_actuator_postions(full_actions)

      # Get the calculated positions of each joint and the end effector
      shoulder_pos = forward_kinematics.fk_shoulder(joint_angles[:3])[:3,3]
      elbow_pos    = forward_kinematics.fk_elbow(joint_angles[:3])[:3,3]
      foot_pos     = forward_kinematics.fk_foot(joint_angles[:3])[:3,3]

      # shoulder_pos = np.array([0,0,0,1])
      # Show the bebug spheres for FK
      p.resetBasePositionAndOrientation(shoulder_sphere_id, posObj=shoulder_pos, ornObj=[0, 0, 0, 1])
      p.resetBasePositionAndOrientation(elbow_sphere_id   , posObj=elbow_pos   , ornObj=[0, 0, 0, 1])
      p.resetBasePositionAndOrientation(foot_sphere_id    , posObj=foot_pos    , ornObj=[0, 0, 0, 1])

      # Show the result in the terminal
      if counter % 20 == 0:
        print(f"\rJoint angles: [{', '.join(f'{q: .3f}' for q in joint_angles[:3])}] | Position: ({', '.join(f'{p: .3f}' for p in foot_pos)})", end='')

app.run(main)