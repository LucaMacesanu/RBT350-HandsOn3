import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis
  

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  wx = axis[0]
  wy = axis[1]
  wz = axis[2]
  cost = np.cos(angle)
  sint = np.sin(angle)

  rot_mat = np.block([
    [cost + (wx**2)*(1-cost), wx*wy*(1-cost)-wz*sint, wy*sint+wx*wz*(1-cost)],
    [wz*sint+wx*wy*(1-cost), cost+(wy**2)*(1-cost), -wx*sint + wy*wz*(1-cost)],
    [-wy*sint+wx*wz*(1-cost), wx*sint + wy*wz*(1-cost), cost + (wz**2)*(1-cost)]
  ])

  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array


  """
  r = rotation_matrix(axis,angle)

  transformation = np.block([
    [r[0,0],r[0,1],r[0,2],v_A[0]],
    [r[1,0],r[1,1],r[1,2],v_A[1]],
    [r[2,0],r[2,1],r[2,2],v_A[2]],
    [0,0,0,1]
  ])
  return transformation

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """
  hip_frame = homogenous_transformation_matrix(axis=[0,0,1], angle=joint_angles[0], v_A=[0,0,0])
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  base_to_hip = fk_hip(joint_angles)
  hip_to_shoulder = homogenous_transformation_matrix(axis=[0,1,0],angle=joint_angles[1],v_A=[0,-HIP_OFFSET,0])
  return np.matmul(base_to_hip,hip_to_shoulder)
 
  
def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame


  """
  base_to_shoulder = fk_shoulder(joint_angles)
  shoulder_to_elbow = homogenous_transformation_matrix(axis=[0,1,0],angle=joint_angles[2],v_A=[0,0,UPPER_LEG_OFFSET])
  return np.matmul(base_to_shoulder,shoulder_to_elbow)


def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """
  base_to_elbow = fk_elbow(joint_angles)
  elbow_to_foot = homogenous_transformation_matrix(axis=[0,1,0],angle=0,v_A=[0,0,LOWER_LEG_OFFSET])
  return np.matmul(base_to_elbow,elbow_to_foot)


