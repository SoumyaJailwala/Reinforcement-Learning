import mujoco
import mujoco.viewer
import pyquaternion as pyq
import numpy as np
#run using mjpython?

# code to get pos and quat:
# import mujoco
# model = mujoco.MjModel.from_xml_path('Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach_CLEANTEST.xml')
# data = mujoco.MjData(model)
# mujoco.mj_forward(model, data)
# ee_id = model.body('hand').id  # Change to your end-effector's name
# print('pos:', data.xpos[ee_id], 'quat:', data.xquat[ee_id])

MOCAP_BODY_INDEX = 0  # usually the first mocap body in the model
ROTATION_ANGLE = 10
POSITION_STEP = 0.01

def rotate_quaternion(quat, axis, angle_degrees):
    angle_rad = np.deg2rad(angle_degrees)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q_rot = pyq.Quaternion(axis=axis, angle=angle_rad)
    return (q * q_rot).elements

def key_callback(key, data):
    """
    Mac laptop keycodes mapping for:
    Arrow keys, WASD for position and IJKL,UO for rotation
    
    Note: You can print unknown keys to discover keycodes:
        print(f"Pressed key code: {key}")
    """
    # Mapping Mac laptop arrow keys (usually the same as standard, but confirm)
    # Common macOS keycodes (may vary, so test and adapt as needed)
    KEY_UP = 126
    KEY_DOWN = 125
    KEY_LEFT = 123
    KEY_RIGHT = 124

    # WASD keys for Y movement (+Y forward/back)
    KEY_W = 13
    KEY_S = 1

    # IJKL for rotation axes
    KEY_I = 34
    KEY_K = 40
    KEY_J = 38
    KEY_L = 37

    # U and O rotate around Z axis (+/-)
    KEY_U = 32
    KEY_O = 31

    # Move Up/Down (Z axis) with arrow up/down
    if key == KEY_UP:
        data.mocap_pos[MOCAP_BODY_INDEX, 2] += POSITION_STEP
    elif key == KEY_DOWN:
        data.mocap_pos[MOCAP_BODY_INDEX, 2] -= POSITION_STEP

    # Move Left/Right (X axis)
    elif key == KEY_LEFT:
        data.mocap_pos[MOCAP_BODY_INDEX, 0] -= POSITION_STEP
    elif key == KEY_RIGHT:
        data.mocap_pos[MOCAP_BODY_INDEX, 0] += POSITION_STEP

    # Move Forward/Backward (Y axis) with W and S
    elif key == KEY_W:
        data.mocap_pos[MOCAP_BODY_INDEX, 1] += POSITION_STEP
    elif key == KEY_S:
        data.mocap_pos[MOCAP_BODY_INDEX, 1] -= POSITION_STEP

    # Rotate around X axis with I (positive) and K (negative)
    elif key == KEY_I:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [1, 0, 0], ROTATION_ANGLE)
    elif key == KEY_K:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [1, 0, 0], -ROTATION_ANGLE)

    # Rotate around Y axis with J (positive) and L (negative)
    elif key == KEY_J:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [0, 1, 0], ROTATION_ANGLE)
    elif key == KEY_L:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [0, 1, 0], -ROTATION_ANGLE)

    # Rotate around Z axis with U (+) and O (-)
    elif key == KEY_U:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [0, 0, 1], ROTATION_ANGLE)
    elif key == KEY_O:
        data.mocap_quat[MOCAP_BODY_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_BODY_INDEX], [0, 0, 1], -ROTATION_ANGLE)

    else:
        # To help debugging, uncomment the next line
        # print(f"Unknown key code pressed: {key}")
        pass

def main():
    model_path = 'Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach_CLEANTEST.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch(model, data) as viewer:
        viewer.set_key_callback(lambda key: key_callback(key, data))
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    main()
