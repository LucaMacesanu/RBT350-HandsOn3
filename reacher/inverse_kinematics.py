import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    # Initialize cost to zero
    angles = forward_kinematics.fk_foot(guess)[:,-1][:3]
    dx =   end_effector_pos[0] - angles[0]
    dy =   end_effector_pos[1] - angles[1]
    dz =   end_effector_pos[2] - angles[2]
    return math.sqrt((dx**2) + (dy**2) + (dz**2))

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    # Initialize Jacobian to zero
    J = np.zeros((3, 3))
    print("deltatype", type(delta))

    # Add your solution here.
    for i in range(3):

        forward_d = joint_angles.copy()
        backward_d = joint_angles.copy()

        forward_d[i] += delta
        backward_d[i] -= delta

        forward_change = forward_kinematics.fk_foot(forward_d) #- forward_kinematics.fk_foot(joint_angles)
        backward_change = forward_kinematics.fk_foot(backward_d) #- forward_kinematics.fk_foot(joint_angles)

        forward_change = forward_change[:,-1][:3]
        backward_change = backward_change[:,-1][:3]


        J[:,i] = np.transpose((forward_change - backward_change) / (2*delta))
    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    # Initialize previous cost to infinity
    previous_cost = np.inf
    # Initialize the current cost to 0.0
    cost = 0.0

    for iters in range(MAX_ITERATIONS):
        guess_pos = forward_kinematics.fk_foot(guess)
        # Calculate the Jacobian matrix using finite differences
        J = calculate_jacobian_FD(guess, PERTURBATION)
        print("J",J)
        # Calculate the residual
        residual = end_effector_pos - guess_pos[:,-1][:3]
        print("residual", residual)
        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse using numpy.linalg.pinv
        J_inv = np.linalg.pinv(J)
        # Take a full Newton step to update the guess for joint angles
        guess += np.dot(J_inv, residual)
        print("guess", guess)
        # cost = # Add your solution here.
        cost = ik_cost(end_effector_pos, guess)
        print("cost", cost)
        print("psudeoJ", J_inv)
        breakpoint
        # Calculate the cost based on the updated guess
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost

    return guess
