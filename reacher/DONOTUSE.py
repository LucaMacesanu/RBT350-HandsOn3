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
import cv2 as cv, numpy as np

flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("ik"          , False, "Whether to control arms through cartesian coordinates(IK) or joint angles")
flags.DEFINE_bool("laser"          , False, "laser control for extra credit")
flags.DEFINE_list("set_joint_angles", [], "List of joint angles to set at initialization.")
FLAGS = flags.FLAGS

KP = 5.0  # Amps/rad
KD = 2.0  # Amps/(rad/s)
MAX_CURRENT = 3  # Amps

UPDATE_DT = 0.01  # seconds

HIP_OFFSET = 0.0335  # meters
L1 = 0.08  # meters
L2 = 0.11  # meters

FIX_X = 0.15 # meters
EMA_ALPHA = 0.1 # less alpha = more smooth

def main(argv):
  if FLAGS.laser:
    PIXEL_O = np.array((0, 0)) # reddest
    PIXEL_U = PIXEL_V = MATRIX = None
    ROBOT_O = np.array([FIX_X, -0.1, 0.1])
    ROBOT_U = np.array([FIX_X, -0.1, -0.1])
    ROBOT_V = np.array([FIX_X, 0.1, 0.1])
    old_xyz = None
  run_on_robot = FLAGS.run_on_robot
  reacher = reacher_sim_utils.load_reacher()

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

  if FLAGS.laser:
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    if not cap.isOpened():
      print('cannot open camera')
      return

  last_circles = None

  # Main loop
  while (True):

    # Whether or not to send commands to the real robot
    enable = False

    # If interfacing with the real robot, handle those communications now
    if run_on_robot:
      hardware_interface.read_incoming_data()
      checkEnableMotors()

    # Determine the direction of data transfer
    real_to_sim = not motor_enabled and run_on_robot

    if FLAGS.laser:
      ret, captured_frame = cap.read()
      output_frame = captured_frame.copy()
      # Convert original image to BGR, since Lab is only available from BGR
      captured_frame_bgr = cv.cvtColor(captured_frame, cv.COLOR_BGRA2BGR)
      # First blur to reduce noise prior to color space conversion
      captured_frame_bgr = cv.medianBlur(captured_frame_bgr, 3)
      # Convert to Lab color space, we only need to check one channel (a-channel) for red here
      captured_frame_lab = cv.cvtColor(captured_frame_bgr, cv.COLOR_BGR2Lab)
      # Threshold the Lab image, keep only the red pixels
      # Possible yellow threshold: [20, 110, 170][255, 140, 215]
      # Possible blue threshold: [20, 115, 70][255, 145, 120]
      captured_frame_lab_red = cv.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
      # Second blur to reduce more noise, easier circle detection
      captured_frame_lab_red = cv.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
      # Use the Hough transform to detect circles in the image
      circles = cv.HoughCircles(captured_frame_lab_red, cv.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)
      #time.sleep(0.1)
      if not ret:
        print("can't receive frame (stream end?). exiting...")
        break
      if cv.waitKey(1) == ord('q'):
        break

    if MATRIX is None:
      PIXEL_U = np.array((output_frame.shape[0], 0)) # reddest
      PIXEL_V = np.array((0, output_frame.shape[1])) # reddest
      MATRIX = np.linalg.inv(np.column_stack((PIXEL_U - PIXEL_O, PIXEL_V - PIXEL_O)))

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
        xyz = slider_values
        xyz[0] = FIX_X
      elif FLAGS.laser:
        xyz = [0, 0, 0]
        # if circles is not None:
        #   circles = np.round(circles[0, :]).astype("int")
        #   cv.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
        #   cv.imshow('frame', output_frame)
        #   return circles
        # else:
        #   return None
        # redness = output_frame / 255
        # redness = redness[:, :, 1] / np.linalg.norm(redness, axis=2)
        # reddest = np.array(np.unravel_index(redness.argmax(), redness.shape))

        #cv.circle(redness, reddest[::-1], radius=10, color=1, thickness=2)
        #cv.imshow('redness', output_frame)
        if circles is not None:
          last_circles = circles

        last_circles = np.round(last_circles[0, :]).astype("int")
        cv.circle(output_frame, center=(last_circles[0, 0], last_circles[0, 1]), radius=last_circles[0, 2], color=(0, 255, 0), thickness=2)
        cv.imshow('frame', output_frame)

        u, v = MATRIX @ ((last_circles[0, 0], last_circles[0, 1]) - PIXEL_O)
        xyz = ROBOT_O + u * (ROBOT_U - ROBOT_O) + v * (ROBOT_V - ROBOT_O)
        xyz[0] = FIX_X
        if old_xyz is not None:
          xyz = xyz * EMA_ALPHA + (1-EMA_ALPHA) * old_xyz
        old_xyz = xyz
      else:
        joint_angles = slider_values
        enable = True

      # If IK is enabled, update joint angles based off of goal XYZ position
      if FLAGS.ik or FLAGS.laser:
        p.resetBasePositionAndOrientation(target_sphere_id, posObj=xyz, ornObj=[0, 0, 0, 1])
        ret = inverse_kinematics.calculate_inverse_kinematics(xyz, joint_angles[:3])
        if ret is not None:
            enable = True
            # Wraps angles between -pi, pi
            joint_angles = np.arctan2(np.sin(ret), np.cos(ret))

            # Double check that the angles are a correct solution before sending anything to the real robot
            pos = forward_kinematics.fk_foot(joint_angles[:3])[:3,3]
            if np.linalg.norm(np.asarray(pos) - xyz) > 0.05:
              # joint_angles = np.zeros_like(joint_angles)
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

      # Show the bebug spheres for FK
      p.resetBasePositionAndOrientation(shoulder_sphere_id, posObj=shoulder_pos, ornObj=[0, 0, 0, 1])
      p.resetBasePositionAndOrientation(elbow_sphere_id   , posObj=elbow_pos   , ornObj=[0, 0, 0, 1])
      p.resetBasePositionAndOrientation(foot_sphere_id    , posObj=foot_pos    , ornObj=[0, 0, 0, 1])

      # Show the result in the terminal
      if counter % 20 == 0:
        print(f"\rJoint angles: [{', '.join(f'{q: .3f}' for q in joint_angles[:3])}] | Position: ({', '.join(f'{p: .3f}' for p in foot_pos)})", end='')

  if FLAGS.laser:
    cap.release()
    cv.destroyAllWindows()

app.run(main)
