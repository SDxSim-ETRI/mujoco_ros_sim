#include "mujoco_ros_sim/mujoco_ros_sim.hpp"
namespace MujocoRosSim
{
    MujocoRosSimNode::MujocoRosSimNode()
    : Node("mujoco_sim_node")
    {
        // parameters
        this->declare_parameter<std::string>("robot_name", "");
        this->declare_parameter<std::string>("model_xml", "");
        this->declare_parameter<double>("camera_fps", 60.0);
        this->declare_parameter<std::string>("controller_class", "");

        // param init
        robot_name_       = this->get_parameter("robot_name").as_string();
        model_xml_        = this->get_parameter("model_xml").as_string();
        camera_fps_       = this->get_parameter("camera_fps").as_double();
        controller_class_ = this->get_parameter("controller_class").as_string();
        if (camera_fps_ <= 0) camera_fps_ = 60.0;

        // publisher
        pub_joint_state_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);

        // version check
        LOGI(this, "MuJoCo version %s\n", mj_versionString());
        if (mjVERSION_HEADER!=mj_version()) 
        {
            LOGE(this, "Headers and library have different versions");
        }

        // plugin scan
        scanPluginLibraries();

        // default setup
        mjv_defaultCamera(&cam_);
        mjv_defaultOption(&opt_);
        mjv_defaultPerturb(&pert_);

        // sim creation
        sim_ = std::make_unique<mj::Simulate>(
            std::make_unique<mj::GlfwAdapter>(),
            &cam_, &opt_, &pert_, /* is_passive = */ false
        );

        // model load
        const std::string xml = resolveModelPath();
        if (xml.empty()) 
        {
            LOGE(this, "Model XML not found (robot_name or model_xml).");
            throw std::runtime_error("model not found");
        }

        // physics thread
        physics_thread_ = std::thread(&MujocoRosSimNode::PhysicsThread, this, sim_.get(), xml);

        // timer setup
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        timer_joint_state_ = this->create_wall_timer(10ms, std::bind(&MujocoRosSimNode::publishJointState, this), cb_group_);

        // shutdown hook
        rclcpp::on_shutdown([this](){ this->requestQuit();});
    }

    MujocoRosSimNode::~MujocoRosSimNode()
    {
        // cleanup
        requestQuit();
        if (control_thread_.joinable()) control_thread_.join();
        if (physics_thread_.joinable()) physics_thread_.join();

        // resource release
        if (sim_) 
        {
            const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
            if (data_)  { mj_deleteData(data_);  data_  = nullptr; }
            if (model_) { mj_deleteModel(model_); model_ = nullptr; }
        }
    }

    std::string MujocoRosSimNode::getExecutableDir() 
    {
        constexpr char kPathSep = '/';
        const char* path = "/proc/self/exe";

        std::string realpath = [&]() -> std::string 
        {
            std::unique_ptr<char[]> realpath(nullptr);
            std::uint32_t buf_size = 128;
            bool success = false;
            while (!success) 
            {
                realpath.reset(new(std::nothrow) char[buf_size]);
                if (!realpath) 
                {
                    std::cerr << "cannot allocate memory to store executable path\n";
                    return "";
                }

                std::size_t written = readlink(path, realpath.get(), buf_size);
                if (written < buf_size) 
                {
                    realpath.get()[written] = '\0';
                    success = true;
                } 
                else if (written == -1) 
                {
                    if (errno == EINVAL) 
                    {
                        // path is already not a symlink, just use it
                        return path;
                    }

                    std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
                    return "";
                } 
                else 
                {
                    // realpath is too small, grow and retry
                    buf_size *= 2;
                }
            }
            return realpath.get();
        }();

        if (realpath.empty()) 
        {
            return "";
        }

        for (std::size_t i = realpath.size() - 1; i > 0; --i) 
        {
            if (realpath.c_str()[i] == kPathSep) 
            {
                return realpath.substr(0, i);
            }
        }

        // don't scan through the entire file system's root
        return "";
    }

    void MujocoRosSimNode::scanPluginLibraries() 
    {
        // check and print plugins that are linked directly into the executable
        int nplugin = mjp_pluginCount();
        if (nplugin) 
        {
            std::printf("Built-in plugins:\n");
            for (int i = 0; i < nplugin; ++i) 
            {
                std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
            }
        }

        const std::string sep = "/";

        // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
        // ${EXECDIR} is the directory containing the simulate binary itself
        // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
        const std::string executable_dir = getExecutableDir();
        if (executable_dir.empty()) 
        {
            return;
        }

        const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
        mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char* filename, int first, int count) 
        {
            std::printf("Plugins registered by library '%s':\n", filename);
            for (int i = first; i < first + count; ++i) 
            {
                std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
            }
        });
    }

    const char* MujocoRosSimNode::Diverged(int disableflags, const mjData* data_) 
    {
        if (disableflags & mjDSBL_AUTORESET) 
        {
            for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) 
            {
                if (data_->warning[w].number > 0) 
                {
                    return mju_warningText(w, data_->warning[w].lastinfo);
                }
            }
        }
        return nullptr;
    }

    mjModel* MujocoRosSimNode::LoadModel(const char* file, mj::Simulate& sim) 
    {
        // this copy is needed so that the mju::strlen call below compiles
        char filename[mj::Simulate::kMaxFilenameLength];
        mju::strcpy_arr(filename, file);

        // make sure filename is not empty
        if (!filename[0])
        {
            return nullptr;
        }

        // load and compile
        char loadError[kErrorLength] = "";
        mjModel* mnew = 0;
        auto load_start = mj::Simulate::Clock::now();

        std::string filename_str(filename);
        std::string extension;
        size_t dot_pos = filename_str.rfind('.');

        if (dot_pos != std::string::npos && dot_pos < filename_str.length() - 1) 
        {
            extension = filename_str.substr(dot_pos);
        }

        if (extension == ".mjb") 
        {
            mnew = mj_loadModel(filename, nullptr);
            if (!mnew) 
            {
                mju::strcpy_arr(loadError, "could not load binary model");
            }
        } 
        else 
        {
            mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

            // remove trailing newline character from loadError
            if (loadError[0]) 
            {
                int error_length = mju::strlen_arr(loadError);
                if (loadError[error_length-1] == '\n') 
                {
                    loadError[error_length-1] = '\0';
                }
            }
        }
        auto load_interval = mj::Simulate::Clock::now() - load_start;
        double load_seconds = Seconds(load_interval).count();

        if (!mnew)
        {
            std::printf("%s\n", loadError);
            mju::strcpy_arr(sim.load_error, loadError);
            return nullptr;
        }

        // compiler warning: print and pause
        if (loadError[0]) 
        {
            // mj_forward() below will print the warning message
            std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
            sim.run = 0;
        }

        // if no error and load took more than 1/4 seconds, report load time
        else if (load_seconds > 0.25) 
        {
            mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
        }

        mju::strcpy_arr(sim.load_error, loadError);

        return mnew;
    }

    void MujocoRosSimNode::PhysicsThread(mj::Simulate* sim, std::string filename) 
    {
        // request loadmodel if file given (otherwise drag-and-drop)
        if (!filename.empty())
        {
            sim->LoadMessage(filename.c_str());
            model_ = LoadModel(filename.c_str(), *sim);
            if (model_) 
            {
                // lock the sim mutex
                const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
                data_ = mj_makeData(model_);
            }
            if (data_) 
            {
                sim->Load(model_, data_, filename.c_str());
                // lock the sim mutex
                const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
                mj_forward(model_, data_);

            } 
            else 
            {
                sim->LoadMessageClear();
            }

            dt_ = model_->opt.timestep;

            buildSlices();
            LOGI(this, PrintTable(robot_name_, model_).c_str());
            is_mujoco_ready_ = true;

            PhysicsLoop(*sim);
        }
    }

    void MujocoRosSimNode::PhysicsLoop(mj::Simulate& sim) 
    {
        // cpu-sim synchronization point
        std::chrono::time_point<mj::Simulate::Clock> syncCPU;
        mjtNum syncSim = 0;

        // run until asked to exit
        while (!sim.exitrequest.load()) 
        {
            if (sim.droploadrequest.load()) 
            {
                is_mujoco_ready_ = false;
                sim.LoadMessage(sim.dropfilename);
                mjModel* mnew = LoadModel(sim.dropfilename, sim);
                sim.droploadrequest.store(false);

                mjData* dnew = nullptr;
                if (mnew) dnew = mj_makeData(mnew);
                if (dnew) {
                sim.Load(mnew, dnew, sim.dropfilename);

                // lock the sim mutex
                const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

                mj_deleteData(data_);
                mj_deleteModel(model_);

                model_ = mnew;
                data_ = dnew;
                mj_forward(model_, data_);
                dt_ = model_->opt.timestep;
                buildSlices();
                is_mujoco_ready_ = true;

                } 
                else 
                {
                    sim.LoadMessageClear();
                }
            }

            if (sim.uiloadrequest.load()) 
            {
                is_mujoco_ready_ = false;
                sim.uiloadrequest.fetch_sub(1);
                sim.LoadMessage(sim.filename);
                mjModel* mnew = LoadModel(sim.filename, sim);
                mjData* dnew = nullptr;
                if (mnew) dnew = mj_makeData(mnew);
                if (dnew) 
                {
                    sim.Load(mnew, dnew, sim.filename);

                    // lock the sim mutex
                    const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

                    mj_deleteData(data_);
                    mj_deleteModel(model_);

                    model_ = mnew;
                    data_ = dnew;
                    mj_forward(model_, data_);
                    dt_ = model_->opt.timestep;
                    buildSlices();
                    is_mujoco_ready_ = true;

                } 
                else 
                {
                    sim.LoadMessageClear();
                }
            }

            // sleep for 1 ms or yield, to let main thread run
            //  yield results in busy wait - which has better timing but kills battery life
            if (sim.run && sim.busywait) 
            {
                std::this_thread::yield();
            } 
            else 
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            {
            // lock the sim mutex
            const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

            // run only if model is present
            if (model_) 
            {
                // running
                if (sim.run) 
                {
                    bool stepped = false;

                    // record cpu time at start of iteration
                    const auto startCPU = mj::Simulate::Clock::now();

                    // elapsed CPU and simulation time since last sync
                    const auto elapsedCPU = startCPU - syncCPU;
                    double elapsedSim = data_->time - syncSim;

                    // requested slow-down factor
                    double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

                    // misalignment condition: distance from target sim time is bigger than syncMisalign
                    bool misaligned =
                    std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > kSyncMisalign;

                    // out-of-sync (for any reason): reset sync times, step
                    if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                    misaligned || sim.speed_changed) 
                    {
                        // re-sync
                        syncCPU = startCPU;
                        syncSim = data_->time;
                        sim.speed_changed = false;

                        // inject noise
                        sim.InjectNoise();

                        // run single step, let next iteration deal with timing
                        mj_step(model_, data_);
                        const char* message = Diverged(model_->opt.disableflags, data_);
                        if (message) 
                        {
                            sim.run = 0;
                            mju::strcpy_arr(sim.load_error, message);
                        }
                        else 
                        {
                            stepped = true;
                        }
                    }

                    // in-sync: step until ahead of cpu
                    else 
                    {
                        bool measured = false;
                        mjtNum prevSim = data_->time;

                        double refreshTime = kSimRefreshFraction/sim.refresh_rate;

                        // step while sim lags behind cpu and within refreshTime
                        while (Seconds((data_->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                        mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) 
                        {
                            // measure slowdown before first step
                            if (!measured && elapsedSim) 
                            {
                                sim.measured_slowdown =
                                std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                                measured = true;
                            }

                            // inject noise
                            sim.InjectNoise();

                            // call mj_step
                            mj_step(model_, data_);
                            const char* message = Diverged(model_->opt.disableflags, data_);
                            if (message) 
                            {
                                sim.run = 0;
                                mju::strcpy_arr(sim.load_error, message);
                            } 
                            else 
                            {
                                stepped = true;
                            }

                            // break if reset
                            if (data_->time < prevSim) 
                            {
                                break;
                            }
                        }
                    }

                    // save current state to history buffer
                    if (stepped) 
                    {
                        sim.AddToHistory();
                    }
                }

                // paused
                else 
                {
                    // run mj_forward, to update rendering and joint sliders
                    mj_forward(model_, data_);
                    if (sim.pause_update)
                    {
                        mju_copy(data_->qacc_warmstart, data_->qacc, model_->nv);
                    }
                    sim.speed_changed = true;
                }
            }
            }  // release std::lock_guard<std::mutex>
        }
    }

    std::string MujocoRosSimNode::resolveModelPath() const 
    {
        if (!model_xml_.empty()) return model_xml_;
        if (!robot_name_.empty()) 
        {
            try 
            {
                const auto share = ament_index_cpp::get_package_share_directory("mujoco_ros_sim");
                return share + "/mujoco_menagerie/" + robot_name_ + "/scene.xml";
            } 
            catch (...) 
            {
            }
        }
        return "";
    }

    void MujocoRosSimNode::buildSlices() 
    {
        // reset containers
        joint_names_.clear(); 
        joint_slices_.clear();
        sensor_slices_.clear();
        image_slices_.clear();

        // thread-safe access
        {
            const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
            if (!model_) return;
    
            // joints enumeration
            for (int i = 0; i < model_->njnt; ++i) 
            {
                const char* nm = mj_id2name(model_, mjOBJ_JOINT, i);
                if (!nm || !*nm) continue;
                std::string jn(nm);
                joint_names_.push_back(jn);
    
                const int idx_q  = model_->jnt_qposadr[i];
                const int idx_v  = model_->jnt_dofadr[i];
                const int next_q = (i+1<model_->njnt)? model_->jnt_qposadr[i+1] : model_->nq;
                const int next_v = (i+1<model_->njnt)? model_->jnt_dofadr[i+1] : model_->nv;
    
                joint_slices_.push_back({idx_q, next_q-idx_q, idx_v, next_v-idx_v, jn});
            }
    
            // actuators mapping
            for (int i = 0; i < model_->nu; ++i)
            {
                const char* nm = mj_id2name(model_, mjOBJ_ACTUATOR, i);
                // const int id = model_->actuator_trnid[2 * i];
                // actName2actID_[nm] = id;
                actName2actID_[nm] = i;
            }
    
            // sensors slicing
            for (int i = 0; i < model_->nsensor; ++i) 
            {
                const int adr = model_->sensor_adr[i];
                const int dim = model_->sensor_dim[i];
                const char* nm = mj_id2name(model_, mjOBJ_SENSOR, i);
                std::string sn = nm ? std::string(nm) : ("sens"+std::to_string(i));
                sensor_slices_.push_back({adr, dim, sn});
            }
    
            // default framebuffer size
            const int def_w = model_->vis.global.offwidth  > 0 ? model_->vis.global.offwidth  : 640;
            const int def_h = model_->vis.global.offheight > 0 ? model_->vis.global.offheight : 480;
    
            // cameras listing
            for (int i = 0; i < model_->ncam; ++i) 
            {
                const char* nm = mj_id2name(model_, mjOBJ_CAMERA, i);
                std::string name = (nm && *nm) ? std::string(nm) : ("camera" + std::to_string(i));
                image_slices_.push_back({ i, def_w, def_h, name });
            }
        }
    }

    void MujocoRosSimNode::ImageThread()
    {
        // wait for hidden window
        while (rclcpp::ok() && !offscreen_ready_.load()) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!offscreen_) return;

        // bind GL context
        glfwMakeContextCurrent(offscreen_);

        // wait for model ready
        while (rclcpp::ok()) 
        {
            {
                const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
                if (is_mujoco_ready_ && model_ && data_ && model_->ncam > 0) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        {
            // create render resources
            const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
            mjv_defaultScene(&scene_);
            mjv_makeScene(model_, &scene_, 10000);
            mjr_defaultContext(&context_);
            mjr_makeContext(model_, &context_, mjFONTSCALE_150);
        }

        // render loop
        ImageLoop();

        {
            // destroy render resources
            const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
            mjr_freeContext(&context_);
            mjv_freeScene(&scene_);
        }
        
        // unbind GL context
        glfwMakeContextCurrent(nullptr);
    }

    void MujocoRosSimNode::ImageLoop()
    {
        // vertical flip
        auto flip = [&](auto& buf, int channels, int width, int height)
        {
            size_t rowBytes = (size_t)width * channels * sizeof(buf[0]);
            std::vector<std::remove_reference_t<decltype(buf[0])>> tmp(width*channels);
            for (int y=0; y<height/2; ++y) 
            {
                auto* rowA = buf.data() + (size_t)y * width * channels;
                auto* rowB = buf.data() + (size_t)(height-1-y) * width * channels;
                std::memcpy(tmp.data(), rowA, rowBytes);
                std::memcpy(rowA, rowB, rowBytes);
                std::memcpy(rowB, tmp.data(), rowBytes);
            }
        };

        // main image loop
        while (!sim_->exitrequest.load()) 
        {
            ImageCVMap imgs;
            imgs.clear();
            for(int i=0; i<image_slices_.size(); i++)
            {
                auto s = image_slices_[i];
                mjvCamera cam = cam_;
                cam.type = mjCAMERA_FIXED;
                cam.fixedcamid = s.cam_id;

                {
                    // update scene (locked)
                    const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
                    mjv_updateScene(model_, data_, &opt_, &pert_, &cam, mjCAT_ALL, &scene_);
                }

                // offscreen render
                mjrRect vp{0, 0, s.width, s.height};
                mjr_setBuffer(mjFB_OFFSCREEN, &context_);
                mjr_render(vp, &scene_, &context_);

                std::vector<unsigned char> rgb(s.width * s.height * 3);
                std::vector<float> depth(s.width * s.height);
                mjr_readPixels(rgb.data(), depth.data(), vp, &context_);

                flip(rgb, 3, s.width, s.height);
                flip(depth, 1, s.width, s.height);

                if(controller_)
                {
                    // pack cv mats
                    RGBDFrame img;
                    cv::Mat rgb_mat(s.height, s.width, CV_8UC3, rgb.data());
                    cv::Mat depth_mat(s.height, s.width, CV_32FC1, depth.data());
                    img.rgb = rgb_mat.clone();
                    img.depth = depth_mat.clone();
                    imgs[s.name] = std::move(img);
                }
            }

            if(controller_)
            {
                // push to controller
                controller_->updateRGBDImage(imgs);
            }

            // throttle by fps
            auto period = std::chrono::duration<double>(1.0 / std::max(1.0, camera_fps_));
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(period));

        }
    }

    void MujocoRosSimNode::publishJointState() 
    {
        // empty check
        if (joint_slices_.empty()) return;

        // msg setup
        sensor_msgs::msg::JointState js;
        js.header.stamp = this->get_clock()->now();
        js.name = joint_names_;

        // read lock
        const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
        if (!model_ || !data_) return;
        for (const auto& sl : joint_slices_) 
        {
            // fill joint data
            for (int k=0; k<sl.nq; ++k) js.position.push_back(data_->qpos[sl.idx_q + k]);
            for (int k=0; k<sl.nv; ++k) js.velocity.push_back(data_->qvel[sl.idx_v + k]);
        }

        // publish
        pub_joint_state_->publish(js);
    }

    bool MujocoRosSimNode::createController(std::string controller_spec)
    {
        if (controller_spec.empty()) 
        {
            LOGE(this, "Controller is empty. Cannot load controller.");
            return false;
        }

        // Optional '@path' → python_path
        std::string python_path;
        if (auto at = controller_spec.find('@'); at != std::string::npos) {
            python_path = controller_spec.substr(at + 1);
            controller_spec.erase(at);
        }

        // ---------- 1) ControllerInterface Load ----------
        try 
        {
            pluginlib::ClassLoader<ControllerInterface> loader("mujoco_ros_sim", "MujocoRosSim::ControllerInterface");

            if (loader.isClassAvailable(controller_spec)) 
            {
                controller_ = loader.createSharedInstance(controller_spec);
                controller_->configure(shared_from_this());
                LOGI(this, "Loaded ControllerInterface plugin: %s", controller_spec.c_str());
            }
        } 
        catch (const std::exception& e) 
        {
            LOGW(this, "Direct ControllerInterface load failed: %s", e.what());
        } 
        catch (...) 
        {
            LOGW(this, "Direct ControllerInterface load failed with unknown error.");
        }

        // ---------- 2) Python fallback ----------
        if (!controller_) 
        {
            try 
            {
                auto slash = controller_spec.find('/');
                if (slash == std::string::npos) 
                {
                    LOGE(this, "Invalid controller_class format. Expected 'pkg/Class' but got '%s'", controller_spec.c_str());
                    return false;
                }

                std::string python_class = controller_spec.substr(0, slash) + ":" + controller_spec.substr(slash + 1);
                std::string plugin_to_load = "mujoco_ros_sim/py_controller";

                // declare parameters if not existing
                if (!this->has_parameter("python_class")) this->declare_parameter<std::string>("python_class", "");
                if (!this->has_parameter("python_path"))  this->declare_parameter<std::string>("python_path", "");

                std::vector<rclcpp::Parameter> params;
                params.emplace_back("python_class", rclcpp::ParameterValue(python_class));
                if (!python_path.empty()) params.emplace_back("python_path", rclcpp::ParameterValue(python_path));
                this->set_parameters(params);

                pluginlib::ClassLoader<ControllerInterface> loader_py("mujoco_ros_sim", "MujocoRosSim::ControllerInterface");

                controller_ = loader_py.createSharedInstance(plugin_to_load);
                controller_->configure(shared_from_this());

                LOGI(this, "Loaded Python-backed ControllerInterface: %s", python_class.c_str());
            } 
            catch (const std::exception& e) 
            {
                LOGE(this, "Failed to load Python controller: %s", e.what());
                return false;
            } 
            catch (...) 
            {
                LOGE(this, "Unknown error while loading Python controller.");
                return false;
            }
        }

        if (!controller_) 
        {
            LOGE(this, "Controller creation failed — controller_ is null after loading (%s)", controller_spec.c_str());
            return false;
        }

        LOGI(this, "Controller successfully loaded.");
        return true;
    }


    void MujocoRosSimNode::initController() 
    { 
        if (createController(controller_class_)) 
        { 
            control_dt_ = controller_->getCtrlTimeStep(); 
            // start controller thread 
            control_thread_ = std::thread(&MujocoRosSimNode::controlLoop, this); 
        } 
    }

    void MujocoRosSimNode::controlLoop()
    {

        // wait for sim readiness
        // TODO: when is_mujoco_ready_ = false, pause
        while (!sim_->exitrequest.load()) 
        {
            {
                const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
                if (model_ && data_ && is_mujoco_ready_) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // early exit guard
        if (!model_ || !data_ || !controller_) 
        {
            LOGE(this, "controlLoop: model_/data_/controller_ is null.");
            return;
        }

        // pre-allocate state maps
        VecMap pos, vel, tau, sensors;
        pos.reserve(joint_slices_.size());
        vel.reserve(joint_slices_.size());
        tau.reserve(joint_slices_.size());

        for (const auto& sl : joint_slices_) 
        {
            pos[sl.name] = Vec::Zero(sl.nq);
            vel[sl.name] = Vec::Zero(sl.nv);
            tau[sl.name] = Vec::Zero(sl.nv);
        }

        for (const auto& ss : sensor_slices_) 
        {
            sensors[ss.name] = Vec::Zero(ss.dim);
        }

        // no actuators warning
        if (actName2actID_.empty()) 
        {
            LOGW(this, "controlLoop: actName2actID_ is empty. No actuator will be updated.");
        }

        // tick scheduling
        auto next_tick = std::chrono::steady_clock::now();
        static bool is_start = true;

        // control loop
        while (rclcpp::ok() && !sim_->exitrequest.load() && is_mujoco_ready_)
        {
            const auto t0 = std::chrono::steady_clock::now();
            double sim_time = 0.0;
            {
                // read MuJoCo state
                const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
                if (!model_ || !data_) break;

                for (const auto& sl : joint_slices_) 
                {
                    for (int k = 0; k < sl.nq; ++k) pos[sl.name](k) = data_->qpos[sl.idx_q + k];
                    for (int k = 0; k < sl.nv; ++k) vel[sl.name](k) = data_->qvel[sl.idx_v + k];
                }
                for (const auto& ss : sensor_slices_) 
                {
                for (int k = 0; k < ss.dim; ++k) sensors[ss.name](k) = data_->sensordata[ss.idx + k];
                }
                sim_time = static_cast<double>(data_->time);
            }
            const auto t1 = std::chrono::steady_clock::now();


            // push state to controller
            controller_->updateState(pos, vel, tau, sensors, sim_time);
            if(is_start)
            {
                // one-shot init
                is_start = false;
                controller_->starting();
            }
            const auto t2 = std::chrono::steady_clock::now();
            // controller compute
            controller_->compute();
            const auto t3 = std::chrono::steady_clock::now();

            // retrieve control map
            CtrlInputMap u_map = controller_->getCtrlInput();
            const auto t4 = std::chrono::steady_clock::now();

            {
                // write controls to MuJoCo
                const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
                if (!model_ || !data_) break;

                for (const auto& [act_name, u_val] : u_map) 
                {
                    auto it = actName2actID_.find(act_name);
                    if (it == actName2actID_.end()) 
                    {
                        LOGW(this, "controlLoop: '%s' not found in actName2actID_. Skipped.", act_name.c_str());
                        continue;
                    }

                    int act_id = it->second;
                    if (act_id < 0 || act_id >= model_->nu) 
                    {
                        LOGW(this,
                        "controlLoop: actName2actID_['%s'] = %d is out of valid actuator index range [0, %d). Skipped.",
                        act_name.c_str(), act_id, model_->nu);
                        continue;
                    }
                    data_->ctrl[act_id] = u_val;
                }
            }
            const auto t5 = std::chrono::steady_clock::now();

            // profiling
            // timing
            const double getData_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double updateState_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            const double compute_ms     = std::chrono::duration<double, std::milli>(t3 - t2).count();
            const double getCtrl_ms     = std::chrono::duration<double, std::milli>(t4 - t3).count();
            const double applyCtrl_ms   = std::chrono::duration<double, std::milli>(t5 - t4).count();
            const double total_ms       = std::chrono::duration<double, std::milli>(t4 - t0).count();

            const double dt_ms   = control_dt_ * 1000.;

            // deadline miss log
            // overrun
            if (dt_ms < total_ms) 
            {
                std::ostringstream oss;
                oss << "\n===================================\n"
                << "getData took "     << std::fixed << std::setprecision(6) << getData_ms     << " ms\n"
                << "updateState took " << std::fixed << std::setprecision(6) << updateState_ms << " ms\n"
                << "compute took "     << std::fixed << std::setprecision(6) << compute_ms     << " ms\n"
                << "getCtrl took "     << std::fixed << std::setprecision(6) << getCtrl_ms     << " ms\n"
                << "applyCtrl took "   << std::fixed << std::setprecision(6) << applyCtrl_ms   << " ms\n"
                << "totalStep took "   << std::fixed << std::setprecision(6) << total_ms       << " ms\n"
                << "===================================";
                LOGW(this, oss.str().c_str());
            }

            // for python controller
            if (auto pyctrl = std::dynamic_pointer_cast<MujocoRosSim::PyController>(controller_)) 
            {
                pyctrl->spinOnce(0.0);
            }


            // rate sleep
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(control_dt_));
            std::this_thread::sleep_until(next_tick);
        }

        // end
        LOGI(this, "controlLoop finished.");
    }

    void MujocoRosSimNode::runRenderLoop()
    {
        if (sim_) sim_->RenderLoop();
    }

    void MujocoRosSimNode::requestQuit()
    {
        if (!sim_) return;
        sim_->exitrequest.store(true);
        glfwPostEmptyEvent();
    }
} // namespace MujocoRosSim

int main(int argc, char** argv)
{
    // ROS2 entry
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MujocoRosSim::MujocoRosSimNode>();

    // controller init
    node->initController();

    // multi-thread executor
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
    exec.add_node(node);


    {
        // hidden offscreen context
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        node->offscreen_ = glfwCreateWindow(64, 64, "hidden_image", nullptr, nullptr);
        if (!node->offscreen_) 
        {
            std::cerr << "failed to create hidden offscreen window\n";
            return 1;
        }
        node->offscreen_ready_.store(true);
    }

    // spin + image threads
    std::thread exec_thread([&](){ exec.spin(); });
    std::thread image_thread([&](){ node->ImageThread(); });  

    // GUI/render loop
    node->runRenderLoop();

    // join threads
    if (image_thread.joinable()) image_thread.join();
    if (exec_thread.joinable())  exec_thread.join();

    // destroy offscreen
    if (node->offscreen_) { glfwDestroyWindow(node->offscreen_); node->offscreen_ = nullptr; }

    // cleanup
    exec.remove_node(node);
    rclcpp::shutdown();
    return 0;
}
