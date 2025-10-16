import serial
import time
import random
import csv
import URBasic
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime


def log_robot_data(csv_writer, stop_event, robot_state):
    while not stop_event.is_set():
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        angle = robot_state['angle']
        height = robot_state['height']
        state = robot_state['state']
        csv_writer.writerow([timestamp, angle, height, state])
        time.sleep(0.3)

def press_and_collect_data(robot, robot_state, target_pos, press_pos, opposite_press_pos):
    states = ['move', 'press', 'hold', 'release', 'press_opposite', 'hold_opposite', 'release_opposite']
    for state in states:
        robot_state['state'] = state
        if state == 'move':
            robot.movel(target_pos, a=0.05, v=0.05)
        elif state == 'press':
            robot.movel(press_pos, a=0.01, v=0.01)
        elif state == 'hold':
            time.sleep(3)  # Hold for 3 seconds
        elif state == 'release':
            robot.movel(target_pos, a=0.05, v=0.05)
        elif state == 'press_opposite':
            robot_state['angle'] = robot_state['angle'] - 180 if robot_state['angle'] > 0 else robot_state['angle'] + 180
            robot.movel(opposite_press_pos, a=0.01, v=0.01)
        elif state == 'hold_opposite':
            time.sleep(3)  # Hold for 3 seconds
        elif state == 'release_opposite':
            robot.movel(target_pos, a=0.05, v=0.05)
        time.sleep(1)


def press_movement(press_distance, angle, target_pos):
    press_vector = -np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    press_vector = press_vector * press_distance * 0.001

    press_pos = target_pos.copy()
    press_pos[0] = target_pos[0] + press_vector[0]
    press_pos[1] = target_pos[1] + press_vector[1]

    return press_pos

def opposite_press_movement(press_distance, angle, target_pos):
    press_vector = -np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    press_vector = press_vector * press_distance * 0.001

    oppo_press_pos = target_pos.copy()
    oppo_press_pos[0] = target_pos[0] - press_vector[0]
    oppo_press_pos[1] = target_pos[1] - press_vector[1]

    return oppo_press_pos


def cylinder_to_pos(angle, height, initial_pos):
    initial_rot_vec = np.array(initial_pos[3:])
    initial_rot = R.from_rotvec(initial_rot_vec)
    rot_z = R.from_euler('z', np.radians(angle))
    new_rot = rot_z * initial_rot
    new_rot_vec = new_rot.as_rotvec()
    new_rot_vec = np.round(new_rot_vec, 3)
    if angle < 0:
        new_rot_vec = abs(new_rot_vec)

    target_pos = initial_pos.copy()
    target_pos[3:] = new_rot_vec
    target_pos[2] = initial_pos[2] - height*0.001

    return target_pos

    

if __name__ == '__main__':

    # It read the robot touch position and time stamp and save it in a csv file
    # The robot will move to a random position and press the sensor

    # Initialize the robot
    ROBOT_IP = '169.254.76.5'
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
    robot.reset_error()
    time.sleep(1)
    print("Initialisation done")

    csv_filename = 'robot_data_0724_600Press.csv'
    
    # Set the initial position of the robot
    hand_cali_pos = [-0.524, -0.126, -0.015, 0, 3.142, 0] # Origin of the sensor
    initial_pos = hand_cali_pos
    initial_pos[2] = initial_pos[2] + 0.05
    robot.movel(initial_pos, a=0.1, v=0.1)
    print("Initial position set")
    time.sleep(2)
    # robot.close()
    
    try:     
    # Open CSV file
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write CSV header
            csv_writer.writerow(['Timestamp', 'Angle', 'Height', 'State'])

            angle = 0
            height = 0

            robot_state = {'angle': 0, 'height': 0, 'state': 'init'}
            stop_event = threading.Event()
            robot_data_thread = threading.Thread(target=log_robot_data, args=(csv_writer, stop_event, robot_state))
            robot_data_thread.start()

            # Repeat the process 100 times
            for _ in range(300):
                # Generate random target position
                angle = random.uniform(-90, 90)  # degrees
                height = random.uniform(3, 50)  # height in mm
                robot_state['angle'] = angle
                robot_state['height'] = height

                target_pos = cylinder_to_pos(angle, height, initial_pos)
                press_distance = 10  # press distance in mm
                press_pos = press_movement(press_distance, angle, target_pos)
                opposite_press_pos = opposite_press_movement(press_distance, angle, target_pos)

                # Collect data during press
                press_and_collect_data(robot, robot_state, target_pos, press_pos, opposite_press_pos)
                # print the current number of presses
                print(f"Presses done: {_+1}")

            stop_event.set()
            robot_data_thread.join()

    except KeyboardInterrupt:
        print("Keyboard interrupt")
        robot.movel(initial_pos, a=0.05, v=0.05)
        robot.close()

    finally:
        robot.movel(initial_pos, a=0.05, v=0.05)

        robot.close()
        print("Done!")