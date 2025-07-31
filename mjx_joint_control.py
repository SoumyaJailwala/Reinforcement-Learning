import sys
import mujoco
import mujoco.viewer
import pyquaternion as pyq
import numpy as np

# run script using mjpython if on mac!!!

# # code to get starting pos and quat:
# import mujoco
# model = mujoco.MjModel.from_xml_path('Metaworld/metaworld/assets/sawyer_xyz/sawyer_push_CLEANED.xml')
# data = mujoco.MjData(model)
# mujoco.mj_forward(model, data)
# ee_id = model.body('hand').id  # Change to your end-effector's name
# print('pos:', data.xpos[ee_id], 'quat:', data.xquat[ee_id])

import mujoco
import mujoco.viewer

MOCAP_BODY_INDEX = 0



def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements



def key_callback(key, data):
    
    # Arrow keys: move in X (left/right), Z (up/down); W/S for Y (fwd/back)
    if key == 126:  # Up
        data.mocap_pos[MOCAP_BODY_INDEX, 2] += 0.01
    elif key == 125:  # Down
        data.mocap_pos[MOCAP_BODY_INDEX, 2] -= 0.01
    elif key == 123:  # Left
        data.mocap_pos[MOCAP_BODY_INDEX, 0] -= 0.01
    elif key == 124:  # Right
        data.mocap_pos[MOCAP_BODY_INDEX, 0] += 0.01
    elif key == 13:   # W
        data.mocap_pos[MOCAP_BODY_INDEX, 1] += 0.01
    elif key == 1:    # S
        data.mocap_pos[MOCAP_BODY_INDEX, 1] -= 0.01

    # I/K/J/L/U/O for rotation (X, Y, Z axes)
    elif key == 34:   # I (rotate +X)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [1,0,0], 10)
    elif key == 40:   # K (rotate -X)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [1,0,0], -10)
    elif key == 38:   # J (rotate +Y)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [0,1,0], 10)
    elif key == 37:   # L (rotate -Y)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [0,1,0], -10)
    elif key == 32:   # U (rotate +Z)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [0,0,1], 10)
    elif key == 31:   # O (rotate -Z)
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_BODY_INDEX], [0,0,1], -10)

    # Uncomment to debug keycodes
    else:
         print(f"Unknown key code: {key}")


def main():
    model = mujoco.MjModel.from_xml_path('Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach_CLEANTEST.xml')
    data = mujoco.MjData(model)

    try:
        viewer = mujoco.viewer.launch(model, data)
        if viewer is None:
            print("🚨 Viewer failed to launch! Try running under X11 (not Wayland), check OpenGL, and VM display settings.")
            sys.exit(1)
        viewer.set_key_callback(lambda key: key_callback(key, data))
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
        viewer.close()
    except Exception as e:
        print("\n🚨 Mujoco viewer failed to launch!")
        print("This is common on macOS with M1/M2 chips due to OpenGL issues.")
        print("Try running on a different machine, using viewer.launch_passive (no keyboard), or dm_control.")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()