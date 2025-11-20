#pragma once

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <filesystem>

#include <mujoco/mujoco.h>
#include "mujoco/glfw_adapter.h"
#include "mujoco/simulate.h"
#include "mujoco/array_safety.h"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "mujoco_ros_sim/utils.hpp"

#include <pluginlib/class_loader.hpp>

#include "mujoco_ros_sim/controller_interface.hpp"
#include "mujoco_ros_sim/py_controller.hpp"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" 
{
    #include <errno.h>
    #include <unistd.h>
}

namespace MujocoRosSim
{

    namespace mj = ::mujoco;
    namespace mju = ::mujoco::sample_util;
    
    using Seconds = std::chrono::duration<double>;
    using namespace std::chrono_literals;
    
    namespace ConsoleColor 
    {
      inline constexpr const char* RESET = "\033[0m";
      inline constexpr const char* BLUE  = "\033[34m"; // Info
      inline constexpr const char* YELLOW= "\033[33m"; // Warn
      inline constexpr const char* RED   = "\033[31m"; // Error
    }
    
    #define LOGI(node, fmt, ...) RCLCPP_INFO((node)->get_logger(),  (std::string(ConsoleColor::BLUE)   + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
    #define LOGW(node, fmt, ...) RCLCPP_WARN((node)->get_logger(),  (std::string(ConsoleColor::YELLOW) + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
    #define LOGE(node, fmt, ...) RCLCPP_ERROR((node)->get_logger(), (std::string(ConsoleColor::RED)    + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)

    
    struct JointSlice { int idx_q{}, nq{}; int idx_v{}, nv{}; std::string name; };
    struct SensorSlice { int idx{}, dim{}; std::string name; };
    struct ImageSlice { int cam_id{}, width{640}, height{480}; std::string name; };
    
    class MujocoRosSimNode : public rclcpp::Node
    {
        public:
            MujocoRosSimNode();           // ctor
            ~MujocoRosSimNode();          // dtor
    
            void initController();        // controller setup
            void runRenderLoop();         // GUI loop
            void ImageThread();           // offscreen thread


            GLFWwindow* offscreen_ = nullptr;          // hidden GLFW window
            std::atomic<bool> offscreen_ready_{false}; // init barrier

        private:
            // ==================================== For MuJoCo Controller =================================================
            // plugin loader 
            std::shared_ptr<ControllerInterface> controller_{nullptr};
            std::string controller_class_;
    
            std::thread control_thread_;  // control worker
            double control_dt_{0.002};    // control period
            bool is_mujoco_ready_{false}; // model ready flag
    
            bool createController(std::string controller_spec);
            void controlLoop();
            // ==================================== For MuJoCo Sim =================================================
            std::string robot_name_;
            std::string model_xml_;
            double dt_{0.002};            // physics timestep
            double camera_fps_{30.0};     // camera rate

            mjvScene     scene_;          // scene graph
            mjrContext   context_;        // render context
            
            rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_state_;
            rclcpp::TimerBase::SharedPtr timer_joint_state_;
            std::shared_ptr<rclcpp::CallbackGroup> cb_group_;
    
            std::vector<std::string> joint_names_;
            std::vector<JointSlice> joint_slices_;
            std::unordered_map<std::string, int> actName2actID_; 
            std::vector<SensorSlice> sensor_slices_;
            std::vector<ImageSlice> image_slices_;

    
            std::string resolveModelPath() const; // model path
            void buildSlices();
            void publishJointState();
            void ImageLoop();
            // ==================================== For MuJoCo =================================================
            // constants
            static constexpr double kSyncMisalign       = 0.1;
            static constexpr double kSimRefreshFraction = 0.7;
            static constexpr int    kErrorLength        = 1024;
            
            // model and data
            mjModel* model_ = nullptr;    // MuJoCo model
            mjData* data_ = nullptr;      // MuJoCo data
    
            mjvCamera cam_;               // view camera
            mjvOption opt_;               // visualization opts
            mjvPerturb pert_;             // mouse perturb
    
            std::unique_ptr<mj::Simulate> sim_; // simulate helper
    
            std::thread physics_thread_;  // physics worker
    
            void requestQuit();           // stop signal
            std::string getExecutableDir();
            void scanPluginLibraries();
            const char* Diverged(int disableflags, const mjData* d);
            mjModel* LoadModel(const char* file, mj::Simulate& sim);
            void PhysicsLoop(mj::Simulate& sim);
            void PhysicsThread(mj::Simulate* sim, std::string filename);
            // =================================================================================================
    
    };

} // namespace MujocoRosSim
