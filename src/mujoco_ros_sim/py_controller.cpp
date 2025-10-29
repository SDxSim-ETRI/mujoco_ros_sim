#include "mujoco_ros_sim/py_controller.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <dlfcn.h>
#include <Python.h>
#include <atomic>
#include <memory>

namespace py = pybind11;
namespace MujocoRosSim
{
    // promote libpython
    static void promote_libpython_to_global() 
    {
        static bool done = false;
        if (done) return;
        done = true;

        const char* cands[] = {
            "libpython3.10.so.1.0",
            "libpython3.10.so",
            nullptr
        };

        for (const char** p = cands; *p; ++p) 
        {
            if (void* h = dlopen(*p, RTLD_NOW | RTLD_GLOBAL)) 
            {
                // global symbols
                return;
            }
        }
        // optional warn
    }

    namespace 
    {
        // init Python
        void ensure_python() 
        {
            static bool inited = false;
            if (!inited) 
            {
                // ::setenv("PYTHONNOUSERSITE", "1", 1);
                py::initialize_interpreter();

                // release GIL
                PyEval_SaveThread();

                inited = true;
            }
        }

        // Eigen → numpy
        py::array eigen_to_np(const Eigen::VectorXd& v) 
        {
            auto a = py::array_t<double>(v.size());
            std::memcpy(a.mutable_data(), v.data(), sizeof(double) * v.size());
            return a;
        }

        // map → dict
        py::dict vecmap_to_pydict(const VecMap& m) 
        {
            py::dict d;
            for (const auto& kv : m) d[py::str(kv.first)] = eigen_to_np(kv.second);
            return d;
        }

        // cv helper
        template<typename T>
        py::array cv_to_np_impl(const cv::Mat& img) 
        {
            const int h = img.rows, w = img.cols, c = img.channels();
            auto out = (c == 1) ? py::array_t<T>({h, w})
            : py::array_t<T>({h, w, c});
            if (img.isContinuous()) 
            {
                std::memcpy(out.mutable_data(), img.data, img.total() * img.elemSize());
            } 
            else 
            {
                cv::Mat tmp = img.clone();
                std::memcpy(out.mutable_data(), tmp.data, tmp.total() * tmp.elemSize());
            }
            return out;
        }

        // cv → numpy
        py::object cv_to_np(const cv::Mat& img) 
        {
            switch (img.depth()) 
            {
                case CV_8U:  return cv_to_np_impl<uint8_t>(img);
                case CV_16U: return cv_to_np_impl<uint16_t>(img);
                case CV_32F: return cv_to_np_impl<float>(img);
                case CV_64F: return cv_to_np_impl<double>(img);
                default:     return cv_to_np_impl<uint8_t>(img);
            }
        }

        // image map → dict
        py::dict imagemap_to_pydict(const ImageCVMap& m)
        {
            py::dict out;
            for (const auto& kv : m)
            {
                const std::string& name = kv.first;
                const RGBDFrame& f = kv.second;

                py::dict pair;
                if (!f.rgb.empty())   pair[py::str("rgb")]   = cv_to_np(f.rgb);
                if (!f.depth.empty()) pair[py::str("depth")] = cv_to_np(f.depth);

                out[py::str(name)] = std::move(pair);
            }
            return out;
        }
    };

    // numpy view
    static py::object make_numpy_view(std::vector<double>& buf) 
    {
        return py::array(py::dtype::of<double>(),
        { (py::ssize_t)buf.size() },
        { (py::ssize_t)sizeof(double) },
        buf.data(),
        /* base = */ py::none());
    }

    // runtime tune
    static void set_low_jitter_runtime_once() 
    {
        static std::once_flag once;
        std::call_once(once, []{
            ::setenv("OMP_NUM_THREADS",       "1", 1);
            ::setenv("OPENBLAS_NUM_THREADS",  "1", 1);
            ::setenv("MKL_NUM_THREADS",       "1", 1);
            ::setenv("NUMEXPR_NUM_THREADS",   "1", 1);

            py::gil_scoped_acquire gil;
            py::module_::import("gc").attr("disable")();
        });
    }

    struct PyController::Impl 
    {
        // python handles
        py::object rclpy_mod, py_node, py_ctrl;
        // bound callables
        py::object fn_spin_once, fn_update_state, fn_compute, fn_get_ctrl, fn_update_img;
    };

    void PyController::configure(const rclcpp::Node::SharedPtr& node)
    {
        ControllerInterface::configure(node);

        std::string py_class = node_->has_parameter("python_class") ? node_->get_parameter("python_class").as_string() : node_->declare_parameter<std::string>("python_class", "");
        std::string py_path = node_->has_parameter("python_path") ? node_->get_parameter("python_path").as_string() : node_->declare_parameter<std::string>("python_path", "");

        if (py_class.empty()) throw std::runtime_error("PyController: parameter 'python_class' must be set (e.g. 'mypkg.mymod:MyController')");

        promote_libpython_to_global();
        ensure_python();

        try 
        {
            py::gil_scoped_acquire gil;

            impl_ = std::make_unique<Impl>();

            // rclpy init
            impl_->rclpy_mod = py::module_::import("rclpy");
            impl_->rclpy_mod.attr("init")();

            // optional sys.path
            if (!py_path.empty()) 
            {
                py::module_ sys = py::module_::import("sys");
                sys.attr("path").cast<py::list>().append(py_path);
            }

            // parse "module:Class" or "module.Class"
            std::string mod, cls;
            {
                auto pos = py_class.find(':');
                if (pos != std::string::npos) 
                { 
                    mod = py_class.substr(0, pos); 
                    cls = py_class.substr(pos+1); 
                }
                else 
                {
                    auto d = py_class.rfind('.');
                    if (d == std::string::npos) throw std::runtime_error("python_class must be 'module:Class' or 'module.Class'");
                    mod = py_class.substr(0, d); cls = py_class.substr(d+1);
                }
            }

            // create rclpy node
            const std::string parent_name = node_->get_name();
            const std::string parent_ns   = node_->get_namespace();
            const std::string py_name     = "py_" + parent_name;
            impl_->py_node = impl_->rclpy_mod.attr("create_node")(
                py::str(py_name),
                py::arg("namespace") = py::str(parent_ns.empty() ? "/" : parent_ns)
            );

            // import controller & bind
            py::object mod_obj = py::module_::import(mod.c_str());
            py::object cls_obj = mod_obj.attr(cls.c_str());
            impl_->py_ctrl       = cls_obj(impl_->py_node);
            impl_->fn_update_state = impl_->py_ctrl.attr("updateState");
            impl_->fn_compute    = impl_->py_ctrl.attr("compute");
            impl_->fn_get_ctrl   = impl_->py_ctrl.attr("getCtrlInput");
            impl_->fn_update_img = impl_->py_ctrl.attr("updateRGBDImage");
            impl_->fn_spin_once    = impl_->rclpy_mod.attr("spin_once");

            try { this->dt_ = impl_->py_ctrl.attr("getCtrlTimeStep")().cast<double>(); } catch (...) {}

            RCLCPP_INFO(node_->get_logger(), "[PyController] loaded python class: %s", py_class.c_str());
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_FATAL(node_->get_logger(), "[PyController:configure] Python error:\n%s", e.what());
            throw;
        }
    }


    void PyController::starting()
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            impl_->py_ctrl.attr("starting")();
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[starting] %s", e.what());
        }
    }

    void PyController::updateState(const VecMap& pos,
                                   const VecMap& vel,
                                   const VecMap& tau_ext,
                                   const VecMap& sensors,
                                   double sim_time)
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            py::dict py_pos = vecmap_to_pydict(pos);
            py::dict py_vel = vecmap_to_pydict(vel);
            py::dict py_tau = vecmap_to_pydict(tau_ext);
            py::dict py_sen = vecmap_to_pydict(sensors);
            impl_->fn_update_state(py_pos, py_vel, py_tau, py_sen, sim_time);
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[updateState] %s", e.what());
        }
    }


    void PyController::updateRGBDImage(const ImageCVMap& images)
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            impl_->fn_update_img(imagemap_to_pydict(images));
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[updateRGBDImage] %s", e.what());
        }
    }


    void PyController::compute()
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            impl_->fn_compute();
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[compute] %s", e.what());
        }
    }


    CtrlInputMap PyController::getCtrlInput() const
    {
        CtrlInputMap out;
        if (!impl_) return out;

        try 
        {
            py::gil_scoped_acquire gil;
            py::dict d = impl_->fn_get_ctrl();
            out.reserve(d.size());
            for (auto item : d) out[item.first.cast<std::string>()] = item.second.cast<double>();
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[getCtrlInput] %s", e.what());
        }
        return out;
    }

    void PyController::spinOnce(double timeout_sec)
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            impl_->fn_spin_once(impl_->py_node, py::arg("timeout_sec") = timeout_sec);
        } 
        catch (const py::error_already_set& e) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[spinOnce] %s", e.what());
        }
    }

    PyController::~PyController()
    {
        if (!impl_) return;
        try 
        {
            py::gil_scoped_acquire gil;
            if (impl_->rclpy_mod) impl_->rclpy_mod.attr("shutdown")();
        } 
        catch (...) {}
    }

    // plugin export
    PLUGINLIB_EXPORT_CLASS(MujocoRosSim::PyController, MujocoRosSim::ControllerInterface)
}// namespace MujocoRosSim