#pragma once
#include "mujoco_ros_sim/controller_interface.hpp"
#include <memory>
#include <mutex>
#include <thread>
#include <cstring>

namespace MujocoRosSim
{
    class PyController final : public ControllerInterface 
    {
        public:
            PyController() = default;
            ~PyController() override;

            void configure(const rclcpp::Node::SharedPtr& node) override;
            void starting() override;
            void updateState(const VecMap& pos,
                             const VecMap& vel,
                             const VecMap& tau_ext,
                             const VecMap& sensors,
                             double sim_time) override;
            void updateRGBDImage(const ImageCVMap& images) override;
            void compute() override;
            CtrlInputMap getCtrlInput() const override;
            
            void spinOnce(double timeout_sec = 0.0);
            
        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
    };
} // namespace MujocoRosSim
