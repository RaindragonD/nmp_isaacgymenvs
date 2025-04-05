"""
This code is kinda messy for compatibility between Dagger and residual RL, TODO: cleanup later
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from typing import Tuple
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.demo_loader import DemoLoader
from isaacgymenvs.utils.torch_jit_utils import *

from utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd
from collections import OrderedDict
from omegaconf import DictConfig
from tqdm import tqdm

from fabrics_sim.fabrics.franka_fabric_rl import FrankaFabricRL
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.worlds.basis_points import BasisPoints
from fabrics_sim.worlds.voxels import VoxelCounter


class FrankaMPFull(FrankaMP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, num_env_per_env=1):
        self.device = sim_device
        self.enable_fabric = cfg["fabric"]["enable"]
        self.force_no_fabric = False
        self.no_base_action = False
        self.vis_basis_points = cfg["fabric"]["vis_basis_points"]
        self.base_policy_only = cfg["env"]["base_policy_only"]

        # Demo loading
        hdf5_path = cfg["env"]["hdf5_path"]
        self.demo_loader = DemoLoader(hdf5_path, cfg["env"]["numEnvs"])
        self.batch_idx = cfg["env"]["batch_idx"]

        # need to change the logic here (2 layers of reset ; multiple start & goal in one env ; relaunch IG)
        self.batch = self.demo_loader.get_next_batch(batch_idx=self.batch_idx)

        self.start_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.goal_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.lock_in = torch.zeros((cfg["env"]["numEnvs"], ), dtype=torch.bool, device=self.device)
        self.lock_in_pos_err = cfg["fabric"]["lock_in_pos_err"] # meters
        self.lock_in_rot_err = cfg["fabric"]["lock_in_rot_err"] # degrees
        self.obstacle_configs = []
        self.obstacle_handles = []
        self.max_obstacles = 0
        self.gt_actions = []

        for env_idx, demo in enumerate(self.batch):
            self.start_config[env_idx] = torch.tensor(demo['states'][0][:7], device=self.device)
            self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)
            gt_actions = torch.tensor(demo['gt_actions'], device=self.device)
            self.gt_actions.append(gt_actions)
            
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + self.max_obstacles  # franka + obstacles
        max_agg_shapes = num_franka_shapes + self.max_obstacles
        self.frankas = []
        self.env_ptrs = []

        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_scene_points = self.pcd_spec_dict['num_obstacle_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        self.combined_pcds = torch.cat(
            (
                torch.zeros(num_robot_points, 4, device=self.device),
                torch.ones(num_scene_points, 4, device=self.device),
                2 * torch.ones(num_target_points, 4, device=self.device),
            ),
            dim=0,
        ).repeat(self.num_envs, 1, 1)


        self.obstacle_count = 0
        self.max_objects_per_env = 20
        self.fabrics_world_dict = dict()

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.objects_per_env = 0

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create obstacles using initial demo data
            env_obstacles = []
            (
                cuboid_dims, 
                cuboid_centers, 
                cuboid_quats,
                cylinder_radii, 
                cylinder_heights,
                cylinder_centers,
                cylinder_quats,
                *_
            ) = self.obstacle_configs[i]

            # num_cylinders = len(cylinder_radii) #pausing cylinders due to incorrect spawning. Likely an actor indexing issue.

            num_cubes = len(cuboid_dims)

            # Create actual obstacles with proper sizes
            for j in range(self.max_obstacles):
                if j < num_cubes:
                    # Create obstacle with actual size and position
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist()
                    )

                    if self.enable_fabric:
                        self._create_fabric_cube(
                            pos=cuboid_centers[j].tolist(),
                            size=cuboid_dims[j].tolist(),
                            quat=cuboid_quats[j].tolist(),
                            env_id=i,
                        )

                else:
                    # Create minimal placeholder obstacles far away
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=[0., 0., -100.0],
                        size=[0.001, 0.001, 0.001],
                        quat=[0, 0, 0, 1]
                    )

                obstacle_actor = self.gym.create_actor(
                    env_ptr,
                    obstacle_asset,
                    obstacle_pose,
                    f"obstacle_{j}",
                    i,
                    1,
                    0
                )
                env_obstacles.append(obstacle_actor)

            # update max_objects_per_envs
            if self.objects_per_env > self.max_objects_per_env:
                self.max_objects_per_env = self.objects_per_env
                
            self.obstacle_handles.append(env_obstacles)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.env_ptrs.append(env_ptr)
            self.frankas.append(franka_actor)

            # compute the static scene pcd (currently only consider static scenes)
            static_scene_pcd = compute_scene_oracle_pcd(
                num_obstacle_points=num_scene_points,
                cuboid_dims=cuboid_dims,
                cuboid_centers=cuboid_centers,
                cuboid_quats=cuboid_quats,
            )
            self.combined_pcds[i, num_robot_points:num_robot_points+num_scene_points, :3] = torch.from_numpy(static_scene_pcd).to(self.device)

        
        if self.enable_fabric:
            self.fabrics_world_model = WorldMeshesModel(
                batch_size=self.num_envs,
                max_objects_per_env=self.max_objects_per_env,
                device=self.device,
                world_dict=self.fabrics_world_dict,
            )
            self.fabrics_object_ids, self.fabrics_object_indicator = self.fabrics_world_model.get_object_ids()

            # Create franka fabric
            self.franka_fabric = FrankaFabricRL(self.num_envs, self.device, "right_gripper")

            # Create integrator for the fabric dynamics.
            self.franka_integrator = DisplacementIntegrator(self.franka_fabric)

            cspace_dim = 7
            self.fabric_q = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
            self.fabric_qd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
            self.fabric_qdd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)

            self.num_points_on_franka = self.franka_fabric.base_fabric_repulsion.num_points
            self.obstacle_dir_robot_frame = torch.zeros((self.num_envs, self.num_points_on_franka, 3), dtype=torch.float, device=self.device)
            self.obstacle_signed_dir_robot_frame = torch.zeros((self.num_envs, self.num_points_on_franka, 3), dtype=torch.float, device=self.device)
            self.obstacle_signed_distances = torch.zeros((self.num_envs, self.num_points_on_franka), dtype=torch.float, device=self.device)

            if self.vis_basis_points:
                # instantiate basis points 
                basis_points_per_dim = np.array([16, 16, 16]) # 16
                # basis_points_per_dim = np.array([20, 20, 20])

                # basis_coord_limits = np.array([[-0.75, -0.75, 0.], [0.75, 0.75, 1.]])
                basis_coord_limits = np.array([[-0.75, -1., -0.1], [1.5, 1., 1.25]]) 
                basis_points = BasisPoints(
                    points_per_dim=basis_points_per_dim,
                    coord_mins=basis_coord_limits[0],
                    coord_maxs=basis_coord_limits[1],
                    object_ids=self.fabrics_object_ids,
                    object_indicator=self.fabrics_object_indicator,
                    device=self.device,
                )

                basis_max_distance = max((basis_coord_limits[1] - basis_coord_limits[0]) / basis_points_per_dim)
                
                # performs the distance queries once since obstacles are static
                basis_points.query()
                # (num_points, 3) 
                self.basis_point_locations = basis_points.points()
                self.basis_point_signed_distances = basis_points.signed_distance()
                self.basis_point_dir_robot_frame = basis_points.direction()


                # Method 1: No local max distance limits
                self.basis_point_signed_distances = torch.clamp(self.basis_point_signed_distances, max=1.0)
                self.basis_point_signed_dir_robot_frame = self.basis_point_dir_robot_frame * self.basis_point_signed_distances.unsqueeze(-1)

                # # Method 2: Local max distance limits
                # exceed_distance_indices = self.basis_point_signed_distances > basis_max_distance
                # self.basis_point_signed_distances[exceed_distance_indices] = 1.0
                # self.basis_point_dir_robot_frame[exceed_distance_indices, :] = 0.0
                # self.basis_point_signed_dir_robot_frame = self.basis_point_dir_robot_frame * self.basis_point_signed_distances.unsqueeze(-1)

        # Setting up voxel counter
        voxel_size = 0.15
        num_voxels_x = 20
        num_voxels_y = 20
        num_voxels_z = 20
        x_min = -0.5
        y_min = -0.75
        z_min = -0.25
        # basis_coord_limits = np.array([[-0.75, -1., -0.1], [1.5, 1., 1.25]])

        self.voxel_counter = VoxelCounter(batch_size=self.num_envs,
                                        device=self.device,
                                        voxel_size=voxel_size,
                                        num_voxels_x=num_voxels_x,
                                        num_voxels_y=num_voxels_y,
                                        num_voxels_z=num_voxels_z,
                                        x_min=x_min,
                                        y_min=y_min,
                                        z_min=z_min)

        self.voxel_visit_binary = torch.zeros((self.num_envs, num_voxels_x*num_voxels_y*num_voxels_z), device=self.device)
        self.num_visited_voxels_t0 = torch.zeros((self.num_envs), device=self.device)

        # Setup data
        actor_num = 1 + self.max_obstacles  # franka  + obstacles
        self.init_data(actor_num=actor_num)

    def _create_fabric_cube(self, pos, size, quat, env_id):
        # TODO: might be better to put this elsewhere
        self.obstacle_count += 1
        self.objects_per_env += 1

        transform = list(pos) + list(quat)
        self.fabrics_world_dict[f"cube_{self.obstacle_count}"] = {
            "env_index": env_id,
            "type": "box",
            "scaling": " ".join(map(str, size)),
            "transform": " ".join(map(str, transform)),
        }
        return

    def _debug_viz_draw(self):
        draw_obstacle_vectors = False

        self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        for i in range(self.num_envs):
            # draw hand frame
            fabric_ee_pose = self.fabric_forward_kinematics(self.states['q'][:, :7])
            px = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()

            py = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()

            pz = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = fabric_ee_pose[:, 0:3][i].cpu().numpy()
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                [0.85, 0.1, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                [0.1, 0.85, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                [0.1, 0.1, 0.85]
            )

            # draw goal frame
            fabric_goal_pose = self.fabric_forward_kinematics(self.goal_config)
            px = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()

            py = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()

            pz = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = fabric_goal_pose[:, 0:3][i].cpu().numpy()
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                [0.85, 0.1, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                [0.1, 0.85, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                [0.1, 0.1, 0.85]
            )

            if draw_obstacle_vectors:
                repulsion_points_robot_frame = self.franka_fabric.get_taskmap_position("body_points").reshape(self.num_envs, self.num_points_on_franka, 3)
                # draw the vectors pointing to the nearest obstacles
                repulsion_pts_robot_frame = repulsion_points_robot_frame[i]
                obstacle_signed_dir_robot_frame = self.obstacle_signed_dir_robot_frame[i]
                nearest_obstacle_pts_robot_frame = repulsion_pts_robot_frame + obstacle_signed_dir_robot_frame
                
                obstacle_signed_dists = self.obstacle_signed_distances[i]

                for j in range(self.num_points_on_franka):
                    p0 = repulsion_pts_robot_frame[j].cpu().numpy()
                    p1 = nearest_obstacle_pts_robot_frame[j].cpu().numpy()
                    dist = obstacle_signed_dists[j].item()
                    self.gym.add_lines(
                        self.viewer, self.env_ptrs[i], 1, 
                        [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]], 
                        # colour gradient where the line goes from green to red as dist approaches zero
                        [1 - dist, dist, 0.0],
                    )

            if self.vis_basis_points:
                # basis point geometry
                sphere_pose = gymapi.Transform()
                sphere_geom = gymutil.WireframeSphereGeometry(0.01, 1, 1, sphere_pose, color=(0, 0, 1))


                # draw the vectors pointing to the nearest obstacles from basis points
                num_basis_points = self.basis_point_locations.shape[0]

                basis_points_signed_dir_robot_frame = self.basis_point_signed_dir_robot_frame[i]
                basis_point_to_obstacle_robot_frame = self.basis_point_locations + basis_points_signed_dir_robot_frame

                # # converts from robot frame to world frame
                # basis_point_locations_world_frame = quat_rotate(world_to_robot_rot, self.basis_point_locations) + world_to_robot_pos
                # basis_point_to_obstacle_world_frame = quat_rotate(world_to_robot_rot, basis_point_to_obstacle_robot_frame) + world_to_robot_pos

                basis_point_signed_dists = self.basis_point_signed_distances[i]

                for j in range(num_basis_points):
                    p0 = self.basis_point_locations[j].cpu().numpy()
                    p1 = basis_point_to_obstacle_robot_frame[j].cpu().numpy()
                    dist = basis_point_signed_dists[j].item()
                    self.gym.add_lines(
                        self.viewer, self.env_ptrs[i], 1,
                        [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]],
                        # colour gradient where the line goes from green to red as dist approaches zero
                        [1 - dist, dist, 0.0],
                    )

                    basis_point_transform = gymapi.Transform()
                    basis_point_transform.p = gymapi.Vec3(*self.basis_point_locations[j])
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env_ptrs[i], basis_point_transform)

    def update_obstacle_configs_from_batch(self, batch_data):
        """Update obstacle configurations from a new batch of demos."""
        self.obstacle_configs = []
        for demo in batch_data:
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)

    def set_robot_joint_state(self, joint_state: torch.Tensor, joint_vel=None, env_ids=None, debug=False):
        super().set_robot_joint_state(joint_state, joint_vel, env_ids=env_ids, debug=debug)
        if self.enable_fabric:
            self.fabric_q[env_ids, :] = torch.clone(joint_state)
            self.fabric_qd[env_ids, :] = torch.zeros_like(joint_state)
            if joint_vel is not None:
                self.fabric_qd[env_ids, :] = torch.clone(joint_vel)
            self.fabric_qdd[env_ids, :] = torch.zeros_like(joint_state)

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.start_config = tensor_clamp(self.start_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_config = tensor_clamp(self.goal_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_ee = self.get_ee_from_joint(self.goal_config)

        self.set_robot_joint_state(self.start_config[env_ids], env_ids=env_ids, debug=False)

        if self.enable_fabric:
            self.fabric_q[env_ids, :] = torch.clone(self.start_config[env_ids])
            self.fabric_qd[env_ids, :] = torch.zeros_like(self.start_config[env_ids])
            self.fabric_qdd[env_ids, :] = torch.zeros_like(self.start_config[env_ids])
        self.voxel_counter.zero_voxels(env_ids)
        self.voxel_visit_binary[env_ids] = torch.zeros_like(self.voxel_visit_binary[env_ids])
        self.num_visited_voxels_t0[env_ids] = torch.zeros_like(self.num_visited_voxels_t0[env_ids])

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.lock_in[env_ids] = False
        self.compute_observations()

    def compute_reward(self, actions=None):
        self.check_robot_collision()
        current_angles = self.get_joint_angles()
        current_ee = self.get_ee_from_joint(current_angles)

        joint_err = torch.norm(current_angles - self.goal_config, dim=1)
        pos_err = torch.norm(current_ee[:, :3] - self.goal_ee[:, :3], dim=1)
        quat_err = orientation_error(self.goal_ee[:, 3:], current_ee[:, 3:])
        self.goal_reaching = (pos_err < self.lock_in_pos_err) & (quat_err < self.lock_in_rot_err)
        self.lock_in[self.goal_reaching] = True
        self.goal_reaching = (pos_err < 0.05) & (quat_err < 15.0) # making it slightly more tolerant atm
        # self.goal_reaching = (pos_err < 0.01) & (quat_err < 15.0) # TODO: should apply this metric later

        num_visited_voxels_t1 = torch.sum(self.voxel_visit_binary, dim=1)

        self.rew_buf[:], self.reset_buf[:], reaching_rewards, intrinsic_rewards = compute_franka_reward(
            self.reset_buf, self.progress_buf,
            joint_err, pos_err, quat_err,
            self.num_visited_voxels_t0, num_visited_voxels_t1,
            self.collision, self.max_episode_length
        )

        self.num_visited_voxels_t0 = num_visited_voxels_t1

        self.extras['reaching_rewards'] = torch.mean(reaching_rewards).item()
        self.extras['intrinsic_rewards'] = torch.mean(intrinsic_rewards).item()
        self.extras['num_visited_voxels_ave'] = torch.mean(num_visited_voxels_t1).item()

        self.success_flags[self.goal_reaching & (self.reset_buf == 1) & (self.collision_flags == 0)] = 1
        self.success_flags[(~self.goal_reaching) & (self.reset_buf == 1)] = 0
        self.reaching_flags[self.goal_reaching & (self.reset_buf == 1)] = 1 # this records reaching rate at the last step, while goal_reaching will be updated each step
        self.reaching_flags[(~self.goal_reaching) & (self.reset_buf == 1)] = 0

        self.extras['success_rate'] = torch.mean(self.success_flags.float()).item()
        self.extras['collision_rate'] = torch.mean(self.collision_flags.float()).item()
        self.extras['reaching_rate'] = torch.mean(self.reaching_flags.float()).item()

        self.collision_flags[self.reset_buf == 1] = 0 # reset collision rate after logging

        if actions is not None:
            self.extras['actions/residual_action_magnitude'] = actions.norm(dim=1).mean()
            self.extras['actions/base_action_magnitude'] = self.base_delta_action.norm(dim=1).mean()

    def compute_fabric_action(self, abs_actions):
        cspace_target = abs_actions[:, 0:7]
        cspace_target[self.lock_in] = self.goal_config[self.lock_in] # if lock in, then switch to pure fabric mode and set the target to the goal
        # cspace_target = self.goal_config # if you want to only use fabric
        gripper_target = self.fabric_forward_kinematics(cspace_target)

        # syncing fabric states with actual sim robot states
        self.fabric_q[:, 0:7] = self.get_joint_angles()
        self.fabric_qd[:, 0:7] = self.states['qd'][:, :7]

        cspace_toggle = torch.ones(self.num_envs, 1, device=self.device)
        self.franka_fabric.set_features(
            gripper_target,
            "quaternion",
            cspace_target,
            cspace_toggle,
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.fabrics_object_ids,
            self.fabrics_object_indicator
        )

        timestep = 1/60.
        # Integrate fabric layer forward at 60 Hz. If policy action rate gets downsampled in the future, 
        # then change the value of 1 below to the downsample factor
        for i in range(1): # TODO: step fabric multiple times so the delta action is not too small
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.franka_integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(), timestep # should be 1/60
            )

        abs_fabric_actions = torch.clone(self.fabric_q[:, 0:7]).contiguous() \
            + 1.0*50/1000 * (self.fabric_qd[:, 0:7]) # This is to ensure that we do not need to set vel target in sim
            # 50/1000 = damping/stiffness of the joint PD gains of the franka arm in sim
            
        # saving final delta actions for dagger
        self.delta_fabric_actions = abs_fabric_actions[:, :7] - self.get_joint_angles()
        return abs_fabric_actions

    def fabric_forward_kinematics(self, q):
        gripper_map = self.franka_fabric.get_taskmap("gripper")
        gripper_pos, _ = gripper_map(q, None)
        
        ee_x_axis_robot_frame = (gripper_pos[:, 3:6] - gripper_pos[:, 0:3])
        ee_x_axis_robot_frame = ee_x_axis_robot_frame / ee_x_axis_robot_frame.norm(p=2, dim=1, keepdim=True)
        ee_y_axis_robot_frame = (gripper_pos[:, 9:12] - gripper_pos[:, 0:3])
        ee_y_axis_robot_frame = ee_y_axis_robot_frame / ee_y_axis_robot_frame.norm(p=2, dim=1, keepdim=True)
        ee_z_axis_robot_frame = (gripper_pos[:, 15:18] - gripper_pos[:, 0:3])
        ee_z_axis_robot_frame = ee_z_axis_robot_frame / ee_z_axis_robot_frame.norm(p=2, dim=1, keepdim=True)

        R = torch.stack((ee_x_axis_robot_frame, ee_y_axis_robot_frame, ee_z_axis_robot_frame), dim=2)
        # convert from wxyz to xyzw
        q = matrix_to_quaternion(R)[:, [1, 2, 3, 0]] 
        ee_pose = torch.cat((gripper_pos[:, 0:3], q), dim=1)
        return ee_pose

    def pre_physics_step(self, actions):
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        current_joint_state = self.get_joint_angles()
        delta_actions = delta_actions * self.action_scale
        self.actions = delta_actions
        # since the below is commonly used for debugging, let's temporarily keep it here
        # self.base_policy_only = True
        # self.enable_fabric = True
        # self.force_no_fabric = False
        if self.base_policy_only:
            abs_actions = current_joint_state + self.base_delta_action
        elif self.no_base_action:
            abs_actions = current_joint_state + delta_actions
        else:
            abs_actions = current_joint_state + delta_actions + self.base_delta_action
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        # vel_targets = torch.zeros_like(abs_actions, device=self.device)
        if self.enable_fabric and (not self.force_no_fabric):
            abs_actions[:, :7] = self.compute_fabric_action(abs_actions)

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))

            # vel_targets[:, :7] = self.fabric_qd[:, 0:7]
            # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

            # update obstacle vectors
            self.obstacle_dir_robot_frame[:, :, :] = self.franka_fabric.base_fabric_repulsion.accel_dir
            self.obstacle_signed_distances[:, :] = self.franka_fabric.base_fabric_repulsion.signed_distance
            # the maximum detectable distance is set to 1
            self.obstacle_signed_distances = torch.clamp(self.obstacle_signed_distances, max=1)
            self.obstacle_signed_dir_robot_frame[:, :, :] = self.obstacle_dir_robot_frame * self.obstacle_signed_distances.unsqueeze(-1)
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))
            # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

    def post_physics_step(self):
        if self.enable_fabric and ((not self.headless) or self.capture_video):
            self._debug_viz_draw()

        # TODO: note, there are differences between fabric fk and direct eef_pos from IG, not sure how much this will affect
        # gripper_pos = self.fabric_forward_kinematics(self.states["q"][:, 0:7])[:, 0:3]
        gripper_pos = self.get_eef_pose()[:, :3]

        # voxel_counts (num_envs, ) -> number of times the voxel at the specified gripper_pos has been visited
        voxel_counts, _ = self.voxel_counter.get_count(gripper_pos)

        # voxels (num_envs, 20*20*20) -> number of times each voxel has been visited
        voxels = self.voxel_counter.voxel_counter.view(self.num_envs, -1)
        self.voxel_visit_binary = torch.where(
            voxels >= 1.,
            1.,
            0.,
        )

        super().post_physics_step()


@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    np.random.seed(0)
    torch.manual_seed(0)
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_task = cfg_dict["task"]
    rl_device = cfg_dict["rl_device"]
    sim_device = cfg_dict["sim_device"]
    headless = cfg_dict["headless"]
    graphics_device_id = 0
    virtual_screen_capture = False
    force_render = False
    env = FrankaMPFull(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        import ipdb ; ipdb.set_trace()
        t1 = time.time()
        env.reset_idx()
        t2 = time.time()

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")

def orientation_error(q1, q2):
    """
    batched orientation error computation
    input shape [B, 4(xyzw)], input ordering doesn't matter
    return absolute difference in degrees
    """
    assert q1.shape == q2.shape, "Desired and current orientations must have the same shape"

    cc = quat_conjugate(q2)
    q_r = quat_mul(q1, cc)

    # Compute the angle difference using the scalar part (w) of q_r
    w = torch.abs(q_r[:, 3])
    err = 2 * torch.acos(torch.clamp(w, -1.0, 1.0)) / torch.pi * 180  # Clamp for numerical stability, return in degrees
    return err

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_franka_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor,
    joint_err: torch.Tensor, pos_err: torch.Tensor, quat_err: torch.Tensor,
    num_visited_voxels_t0: torch.Tensor, num_visited_voxels_t1: torch.Tensor,
    collision_status: torch.Tensor,
    max_episode_length: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # Sparse reaching reward (TODO: change to use key points)
    reaching_rewards = 50*torch.exp(-10*joint_err)

    # intrinsic reward
    intrinsic_rewards = num_visited_voxels_t1 - num_visited_voxels_t0

    rewards = reaching_rewards + intrinsic_rewards

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, reaching_rewards, intrinsic_rewards


if __name__ == "__main__":
    launch_test()
