/**
 * @file tracing_plan.cpp
 * @brief Example using Trajopt for constrained cartesian planning
 *
 * @author Masaki Murooka
 * @date Sep 27, 2018
 * @version TODO
 * @bug No known bugs
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ros/ros.h>
#include <srdfdom/model.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <trajopt/plot_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_utils/config.hpp>
#include <trajopt_utils/logging.hpp>
#include <urdf_parser/urdf_parser.h>
#include <geometry_msgs/PoseArray.h>
#include <eigen_conversions/eigen_msg.h>


// For loading the pose file from a local package
#include <fstream>
#include <ros/package.h>

using namespace trajopt;
using namespace tesseract;

const std::string ROBOT_DESCRIPTION_PARAM = "robot_description"; /**< Default ROS parameter for robot description */
const std::string ROBOT_SEMANTIC_PARAM = "robot_description_semantic"; /**< Default ROS parameter for robot
                                                                          description */

int steps_per_distance_ = 5;
bool plotting_ = false;
urdf::ModelInterfaceSharedPtr urdf_model_; /**< URDF Model */
srdf::ModelSharedPtr srdf_model_;          /**< SRDF Model */
tesseract_ros::KDLEnvPtr env_;             /**< Trajopt Basic Environment */

ros::Publisher cart_traj_pub_;

std::string move_group_name_;
std::string target_link_name_;

ProblemConstructionInfo cppMethod()
{
  ProblemConstructionInfo pci(env_);

  // Populate Basic Info
  pci.basic_info.manip = move_group_name_;
  pci.basic_info.start_fixed = false;
  // pci.opt_info.max_iter = 200;
  // pci.opt_info.min_approx_improve = 1e-3;
  // pci.opt_info.min_trust_box_size = 1e-3;

  // Create Kinematic Object
  pci.kin = pci.env->getManipulator(pci.basic_info.manip);

  // Populate Constraints
  vector<Eigen::Vector3d> key_point_list // point list for Capital N
    = {Eigen::Vector3d(0.3, -0.4, 0.2),
       Eigen::Vector3d(0.3, -0.4, 0.6),
       Eigen::Vector3d(0.3, -0.1, 0.2),
       Eigen::Vector3d(0.3, -0.1, 0.6)};
  unsigned int n_steps = 0;
  for (unsigned int k = 0; k < key_point_list.size() - 1; ++k) {
    Eigen::Vector3d start_point = key_point_list[k];
    Eigen::Vector3d end_point = key_point_list[k+1];
    double distance = (start_point - end_point).norm();
    unsigned int steps = int(distance * steps_per_distance_);
    for (unsigned int i = 0; i < steps; ++i) {
      double ratio = double(i) / (steps - 1);
      std::shared_ptr<StaticPoseCostInfo> pose = std::shared_ptr<StaticPoseCostInfo>(new StaticPoseCostInfo);
      pose->term_type = TT_CNT;
      pose->name = "waypoint_cart_" + std::to_string(i);
      pose->timestep = n_steps;
      pose->link = target_link_name_;
      pose->xyz = start_point * (1.0 - ratio) + end_point * ratio;
      pose->wxyz = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
      pose->pos_coeffs = Eigen::Vector3d(10, 10, 10);
      pose->rot_coeffs = Eigen::Vector3d(10, 10, 10);

      pci.cnt_infos.push_back(pose);
      n_steps++;
    }
  }

  // Set Number of Steps
  pci.basic_info.n_steps = n_steps;

  // Populate Cost Info
  unsigned int num_joints = pci.kin->numJoints();

  std::shared_ptr<JointVelCostInfo> joint_vel = std::shared_ptr<JointVelCostInfo>(new JointVelCostInfo);
  joint_vel->coeffs = std::vector<double>(num_joints, 1.0); // <= depends on joint_number
  joint_vel->name = "joint_vel";
  joint_vel->term_type = TT_COST;
  pci.cost_infos.push_back(joint_vel);

  std::shared_ptr<JointAccCostInfo> joint_acc = std::shared_ptr<JointAccCostInfo>(new JointAccCostInfo);
  joint_acc->coeffs = std::vector<double>(num_joints, 2.0);
  joint_acc->name = "joint_acc";
  joint_acc->term_type = TT_COST;
  pci.cost_infos.push_back(joint_acc);

  std::shared_ptr<JointJerkCostInfo> joint_jerk = std::shared_ptr<JointJerkCostInfo>(new JointJerkCostInfo);
  joint_jerk->coeffs = std::vector<double>(num_joints, 5.0);
  joint_jerk->name = "joint_jerk";
  joint_jerk->term_type = TT_COST;
  pci.cost_infos.push_back(joint_jerk);

  std::shared_ptr<CollisionCostInfo> collision = std::shared_ptr<CollisionCostInfo>(new CollisionCostInfo);
  collision->name = "collision";
  collision->term_type = TT_COST;
  collision->continuous = false;
  collision->first_step = 0;
  collision->last_step = pci.basic_info.n_steps - 1;
  collision->gap = 1;
  collision->info = createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 20);
  pci.cost_infos.push_back(collision);

  // Populate Init Info
  Eigen::VectorXd start_pos = pci.env->getCurrentJointValues(pci.kin->getName());
  pci.init_info.type = InitInfo::STATIONARY;
  // pci.init_info.type = InitInfo::GIVEN_TRAJ;
  pci.init_info.data = start_pos.transpose().replicate(pci.basic_info.n_steps, 1);

  // publish PoseArray of cartesian trajectory
  // geometry_msgs::PoseArray pose_array_msg;
  // pose_array_msg.header.frame_id = "world";
  // pose_array_msg.header.stamp = ros::Time::now();
  // for (auto i = 0; i < pci.basic_info.n_steps; ++i)
  // {
  //   geometry_msgs::Pose pose_msg;
  //   tf::poseEigenToMsg(tool_poses[i], pose_msg);
  //   pose_array_msg.poses.push_back(pose_msg);
  // }
  // cart_traj_pub_.publish(pose_array_msg);

  ROS_INFO("=== ProblemConstructionInfo ===");
  ROS_INFO("Steps per distance: %d", steps_per_distance_);
  ROS_INFO("Number of steps: %d", pci.basic_info.n_steps);
  ROS_INFO("Movegroup: %s", pci.kin->getName().c_str());
  ROS_INFO("Target link: %s", target_link_name_.c_str());
  ROS_INFO("Number of joints: %d", num_joints);

  return pci;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tracing_plan");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;

  // Initial setup
  std::string urdf_xml_string, srdf_xml_string;
  nh.getParam(ROBOT_DESCRIPTION_PARAM, urdf_xml_string);
  nh.getParam(ROBOT_SEMANTIC_PARAM, srdf_xml_string);

  urdf_model_ = urdf::parseURDF(urdf_xml_string);
  srdf_model_ = srdf::ModelSharedPtr(new srdf::Model);
  srdf_model_->initString(*urdf_model_, srdf_xml_string);
  env_ = tesseract_ros::KDLEnvPtr(new tesseract_ros::KDLEnv);
  assert(urdf_model_ != nullptr);
  assert(env_ != nullptr);

  bool success = env_->init(urdf_model_, srdf_model_);
  assert(success);

  // Publisher
  cart_traj_pub_ = nh.advertise<geometry_msgs::PoseArray>("target_cartesian_trajectory", 1, true);

  // Create plotting tool
  tesseract_ros::ROSBasicPlottingPtr plotter(new tesseract_ros::ROSBasicPlotting(env_));

  // Get ROS Parameters
  pnh.param("move_group", move_group_name_, move_group_name_);
  pnh.param("target_link", target_link_name_, target_link_name_);
  pnh.param<int>("steps_per_distance", steps_per_distance_, steps_per_distance_);
  pnh.param("plotting", plotting_, plotting_);

  // Set the robot initial state
  std::unordered_map<std::string, double> ipos;
  // ipos["CHEST_JOINT0"] = 0;
  ipos["RARM_JOINT0"] = -0.010472;
  ipos["RARM_JOINT1"] = 0;
  ipos["RARM_JOINT2"] = -1.74533;
  ipos["RARM_JOINT3"] = 0.26529;
  ipos["RARM_JOINT4"] = 0.164061;
  ipos["RARM_JOINT5"] = 0.055851;
  ipos["LARM_JOINT0"] = 0.010472;
  ipos["LARM_JOINT1"] = 0;
  ipos["LARM_JOINT2"] = -1.74533;
  ipos["LARM_JOINT3"] = -0.26529;
  ipos["LARM_JOINT4"] = 0.164061;
  ipos["LARM_JOINT5"] = -0.055851;

  env_->setState(ipos);

  plotter->plotScene();

  // Set Log Level
  gLogLevel = util::LevelInfo;

  // Setup Problem
  ProblemConstructionInfo pci = cppMethod();
  TrajOptProbPtr prob = ConstructProblem(pci);

  // Solve Trajectory
  ROS_INFO("tracing plan");

  std::vector<tesseract::ContactResultMap> collisions;
  ContinuousContactManagerBasePtr manager = prob->GetEnv()->getContinuousContactManager();

  bool found = tesseract::continuousCollisionCheckTrajectory(
      *manager, *prob->GetEnv(), *prob->GetKin(), prob->GetInitTraj(), collisions);

  ROS_INFO((found) ? ("Initial trajectory is in collision") : ("Initial trajectory is collision free"));

  BasicTrustRegionSQP opt(prob);
  opt.setParameters(pci.opt_info);
  if (plotting_)
  {
    opt.addCallback(PlotCallback(*prob, plotter));
  }

  opt.initialize(trajToDblVec(prob->GetInitTraj()));
  ros::Time tStart = ros::Time::now();
  sco::OptStatus status = opt.optimize();
  ROS_INFO("Optimization Status: %s, Planning time: %.3f",
           sco::statusToString(status).c_str(),
           (ros::Time::now() - tStart).toSec());

  if (plotting_)
  {
    plotter->clear();
  }

  // Plot the final trajectory
  plotter->plotTrajectory(prob->GetKin()->getJointNames(), getTraj(opt.x(), prob->GetVars()));

  collisions.clear();
  found = tesseract::continuousCollisionCheckTrajectory(
      *manager, *prob->GetEnv(), *prob->GetKin(), prob->GetInitTraj(), collisions);

  ROS_INFO((found) ? ("Final trajectory is in collision") : ("Final trajectory is collision free"));
}
