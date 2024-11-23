import numpy as np
from modern_robotics import CartesianTrajectory


def main():
    T_sb = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0963], [0, 0, 0, 1]])

    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    M_0e_home = np.array(
        [[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]]
    )

    # Compute initial end-effector pose in space frame
    T_se_initial = T_sb @ T_b0 @ M_0e_home

    # Cube configurations: initial and goal
    T_sc_initial = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.025], [0, 0, 0, 1]]
    )

    T_sc_goal = np.array([[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, 0.025], [0, 0, 0, 1]])

    # Gripper configurations: grasp and standoff
    T_ce_grasp = np.array(
        [
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
            [0, 1, 0, 0],
            [-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0],
            [0, 0, 0, 1],
        ]
    )

    T_ce_standoff = np.array(
        [
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
            [0, 1, 0, 0],
            [-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0.15],
            [0, 0, 0, 1],
        ]
    )
    k = 1

    # Generate the trajectory
    trajectory = TrajectoryGenerator(
        T_se_initial, T_sc_initial, T_sc_goal, T_ce_grasp, T_ce_standoff, k
    )

    # Write the trajectory to a csv file
    # chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state
    with open("trajectory.csv", "w") as f:
        for row in trajectory:
            f.write(",".join(map(str, row)) + "\n")

def TrajectoryGenerator(
    T_se_initial, T_sc_initial, T_sc_final, T_ce_grasp, T_ce_standoff, k
):
    """
    Generate the reference trajectory for the end-effector frame {e}.

    This function generates a trajectory consisting of eight concatenated trajectory segments
    for the end-effector frame. Each segment starts and ends at rest, and the end-effector
    performs specific motions such as moving to a standoff position, grasping, and transporting
    a cube.

    Parameters
    ----------
    T_se_initial : numpy.ndarray
        The initial configuration of the end-effector in the reference trajectory.
        A 4x4 homogeneous transformation matrix.

    T_sc_initial : numpy.ndarray
        The cube's initial configuration. A 4x4 homogeneous transformation matrix.

    T_sc_final : numpy.ndarray
        The cube's desired final configuration. A 4x4 homogeneous transformation matrix.

    T_ce_grasp : numpy.ndarray
        The end-effector's configuration relative to the cube when grasping the cube.
        A 4x4 homogeneous transformation matrix.

    T_ce_standoff : numpy.ndarray
        The end-effector's standoff configuration above the cube, relative to the cube.
        A 4x4 homogeneous transformation matrix.

    k : int
        The number of trajectory reference configurations per 0.01 seconds.
        An integer with a value of 1 or greater. A larger `k` corresponds to higher
        trajectory resolution and smoother motion.

    Returns
    -------
    trajectory : list of list
        A list of lists, where each inner list contains the SE(3) matrix elements
        (row-major order) followed by the gripper state.
    """
    Traj = []
    Tse_init_standoff = T_sc_initial @ T_ce_standoff
    Tse_init_grasp = T_sc_initial @ T_ce_grasp
    Tse_fin_standoff = T_sc_final @ T_ce_standoff
    Tse_fin_grasp = T_sc_final @ T_ce_grasp

    Tf = 3  # Time for each segment
    method = 5  # Quintic time-scaling

    def add_to_traj(T_matrices, gripper_state):
        """
        Converts SE(3) matrices to the trajectory format and appends to Traj.
        """
        for T in T_matrices:
            Traj.append(
                [
                    T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2],
                    T[2, 0], T[2, 1], T[2, 2], T[0, 3], T[1, 3], T[2, 3],
                    gripper_state
                ]
            )

    # Step 1: Move to initial standoff position
    N = max(2, int(k * Tf / 0.01))  # Ensure N > 1
    traj1 = CartesianTrajectory(T_se_initial, Tse_init_standoff, Tf, N, method)
    add_to_traj(traj1, gripper_state=0)

    # Step 2: Move to initial grasp position
    traj2 = CartesianTrajectory(Tse_init_standoff, Tse_init_grasp, Tf, N, method)
    add_to_traj(traj2, gripper_state=0)

    # Step 3: Close the gripper
    traj3 = [Tse_init_grasp] * N
    add_to_traj(traj3, gripper_state=1)

    # Step 4: Move back to initial standoff position
    traj4 = CartesianTrajectory(Tse_init_grasp, Tse_init_standoff, Tf, N, method)
    add_to_traj(traj4, gripper_state=1)

    # Step 5: Move to the final standoff position
    traj5 = CartesianTrajectory(Tse_init_standoff, Tse_fin_standoff, Tf, N, method)
    add_to_traj(traj5, gripper_state=1)

    # Step 6: Move to final grasp position
    traj6 = CartesianTrajectory(Tse_fin_standoff, Tse_fin_grasp, Tf, N, method)
    add_to_traj(traj6, gripper_state=1)

    # Step 7: Open the gripper
    traj7 = [Tse_fin_grasp] * N
    add_to_traj(traj7, gripper_state=0)

    # Step 8: Move back to the final standoff position
    traj8 = CartesianTrajectory(Tse_fin_grasp, Tse_fin_standoff, Tf, N, method)
    add_to_traj(traj8, gripper_state=0)

    return Traj


if __name__ == "__main__":
    main()