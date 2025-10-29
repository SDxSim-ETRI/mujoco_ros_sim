// mujoco_ros_sim/controller_interface.hpp
#pragma once
#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace MujocoRosSim 
{
    struct RGBDFrame 
    {
        cv::Mat rgb;
        cv::Mat depth;
    };

    using Vec          = Eigen::VectorXd;
    using VecMap       = std::unordered_map<std::string, Vec>;
    using CtrlInputMap = std::unordered_map<std::string, double>;
    using ImageCVMap   = std::unordered_map<std::string, RGBDFrame>;

    class ControllerInterface 
    {
        public:
            ControllerInterface() = default;
            virtual ~ControllerInterface() = default;

            virtual void configure(const rclcpp::Node::SharedPtr& node) { node_ = node; }

            virtual void starting() = 0;

            virtual void updateState(const VecMap& pos,
                                     const VecMap& vel,
                                     const VecMap& tau_ext,
                                     const VecMap& sensors,
                                     double sim_time) = 0;

            virtual void updateRGBDImage(const ImageCVMap& images) {}

            virtual void compute() = 0;

            virtual CtrlInputMap getCtrlInput() const = 0;

            double getCtrlTimeStep() const { return dt_; }

        protected:
            rclcpp::Node::SharedPtr node_;
            double dt_{0.001};

        private:
            std::atomic_bool running_{true};
    };
} // namespace MujocoRosSim
