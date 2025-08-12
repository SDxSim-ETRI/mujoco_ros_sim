#!/usr/bin/env python3
import time, threading
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Image
from mujoco_ros_sim_msgs.msg import JointDict, SensorDict, CtrlDict, ImageDict, NamedImage
from array import array

import mujoco
import mujoco.viewer

from mujoco_ros_sim.utils import load_mj_model, print_table, to_NamedFloat64ArrayMsg, to_TimeMsg


    
class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        
        # Parameters/Qos
        desc = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('robot_name', descriptor=desc)
        self.declare_parameter('camera_fps', 60.0, descriptor=desc)
        
        robot_name      = self.get_parameter('robot_name').get_parameter_value().string_value
        self.camera_fps = self.get_parameter('camera_fps').get_parameter_value().double_value

        qos = QoSProfile(reliability= ReliabilityPolicy.BEST_EFFORT,
                         history    = HistoryPolicy.KEEP_LAST,
                         depth      = 1,
                         durability = DurabilityPolicy.VOLATILE)

        # ROS2 publisher & subscriber
        self.joint_dict_pub   = self.create_publisher(JointDict,  'mujoco_ros_sim/joint_dict',  qos)
        self.sensor_dict_pub  = self.create_publisher(SensorDict, 'mujoco_ros_sim/sensor_dict', qos)
        self.image_pub        = self.create_publisher(ImageDict,  'mujoco_ros_sim/image_dict',  qos)
        self.joint_state_pub  = self.create_publisher(JointState, 'joint_states', 10)
        ctrl_command_sub = self.create_subscription(CtrlDict, 'mujoco_ros_sim/ctrl_dict', self.sub_ctrl_cb, qos)

        # MuJoCo init
        self.mj_model = load_mj_model(robot_name)
        self.get_logger().info("\033[1;34m\n" + print_table(robot_name, self.mj_model) + "\033[0m")
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        
        #  Passive viewer
        self.viewer_fps = 60.0
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False)

        # Joint/Actuator/Sensor/Camera dicts
        # - Joint dict 
        self.joint_dict = {"joint_names": [], "jname_to_jid": {}}
        for i in range(self.mj_model.njnt):
            name_adr = self.mj_model.name_jntadr[i]
            jname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            if not jname:
                continue
            self.joint_dict["joint_names"].append(jname)
            self.joint_dict["jname_to_jid"][jname] = i

        # - Actuator dict 
        self.actuator_dict = {"actuator_names": [], "aname_to_aid": {}}
        for i in range(self.mj_model.nu):
            name_adr = self.mj_model.name_actuatoradr[i]
            aname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.actuator_dict["actuator_names"].append(aname)
            self.actuator_dict["aname_to_aid"][aname] = i

        # - Sensor dict 
        self.sensor_dict = {"sensor_names": [], "sname_to_sid": {}, "sname_to_sdim": {}}
        for i in range(self.mj_model.nsensor):
            name_adr = self.mj_model.name_sensoradr[i]
            sname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.sensor_dict["sensor_names"].append(sname)
            self.sensor_dict["sname_to_sid"][sname] = i
            self.sensor_dict["sname_to_sdim"][sname] = self.mj_model.sensor_dim[name_adr]

        # - Camera dict 
        self.camera_dict = {"cam_names": [], "cname_to_cid": {}, "resolution": (0, 0)}
        for i in range(self.mj_model.ncam):
            cadr  = self.mj_model.name_camadr[i]
            cname = self.mj_model.names[cadr:].split(b'\x00', 1)[0].decode('utf-8')
            if cname in ("", "free"):                         # skip the free cam
                continue
            self.camera_dict["cam_names"].append(cname)
            self.camera_dict["cname_to_cid"][cname] = cadr
        
        offw = int(self.mj_model.vis.global_.offwidth)  or 640
        offh = int(self.mj_model.vis.global_.offheight) or 480
        self.camera_dict["resolution"] = (offw, offh)
        
        # Slices indicies
        # - Joint sloces
        self._j_slices = []  # (idx_q, nq, idx_v, nv, jname)
        for jname in self.joint_dict["joint_names"]:
            jid   = self.joint_dict["jname_to_jid"][jname]
            idx_q = self.mj_model.jnt_qposadr[jid]
            idx_v = self.mj_model.jnt_dofadr[jid]
            next_q = (self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq)
            next_v = (self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv)
            nq = next_q - idx_q
            nv = next_v - idx_v
            self._j_slices.append((idx_q, nq, idx_v, nv, jname))
        #  - Sensor slices
        self._s_slices = []  # (idx_s, ns, sname)
        for sname in self.sensor_dict["sensor_names"]:
            idx_s = self.sensor_dict["sname_to_sadr"][sname]
            ns    = self.sensor_dict["sname_to_sdim"][sname]
            self._s_slices.append((idx_s, ns, sname))

        # Buffers msgs/images
        # joint/sensor messages (updated by list reference)
        self._pos_msgs = [to_NamedFloat64ArrayMsg(jn, np.zeros(nq)) for (_, nq,  _, _,  jn) in self._j_slices]
        self._vel_msgs = [to_NamedFloat64ArrayMsg(jn, np.zeros(nv)) for (_,  _,  _, nv, jn) in self._j_slices]
        self._tau_msgs = [to_NamedFloat64ArrayMsg(jn, np.zeros(nv)) for (_,  _,  _, nv, jn) in self._j_slices]
        self._sensor_msgs = [to_NamedFloat64ArrayMsg(sn, np.zeros(ns)) for (_, ns, sn) in self._s_slices]
        
        # keep references to the underlying Python lists for in-place update
        self._pos_lists    = [m.value.data for m in self._pos_msgs]
        self._vel_lists    = [m.value.data for m in self._vel_msgs]
        self._tau_lists    = [m.value.data for m in self._tau_msgs]
        self._sensor_lists = [m.value.data for m in self._sensor_msgs]
        
        # camera message bundle (bytearray zero-copy for RGB)
        self._imgdict_msg = ImageDict()
        self._imgdict_msg.images = []
        self._cam_buf_views = {}     # cname -> np.frombuffer(view of bytearray)
        self._cam_named = {}         # cname -> NamedImage (for O(1) access)

        if len(self.camera_dict["cam_names"]) > 0:
            W, H = self.camera_dict["resolution"]
            for cname in self.camera_dict["cam_names"]:
                img = Image()
                img.width, img.height = W, H
                img.encoding = "rgb8"
                img.is_bigendian = 0
                img.step = W * 3
                img.data = bytearray(W * H * 3)
                self._cam_buf_views[cname] = np.frombuffer(img.data, dtype=np.uint8)
                ni = NamedImage()
                ni.name, ni.image = cname, img
                self._imgdict_msg.images.append(ni)
                self._cam_named[cname] = ni
 
        # Concurrency locks/threads
        self._mj_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self._render_data = mujoco.MjData(self.mj_model)
        
        # Timers loops
        cb = ReentrantCallbackGroup()
        self.create_timer(self.dt,               self.sim_loop,              callback_group=cb)
        self.create_timer(1.0 / self.viewer_fps, self.viewer_loop,           callback_group=cb)
        self.create_timer(0.01,                  self.pub_joint_cb, callback_group=cb)
        
        # Caera thread
        # Camera on dedicated thread — not create_timer:
        # 1) GL 컨텍스트 스레드 친화도: Renderer는 생성된 그 스레드에서만 안전(타이머는 워커 스레드가 매번 바뀜).
        # 2) FPS 안정성: executor 지연/지터와 독립적으로 perf_counter 기반 페이싱 유지.
        # 3) 락 경합 감소: _mj_lock은 상태 스냅샷에만 짧게, mj_forward/render는 락 밖에서 수행.
        # 4) 종료 경합 회피: 퍼블리셔 파괴 중 publish 방지(가드) + thread join으로 InvalidHandle 예방.
        # 참고) 타이머를 쓰려면 전용 SingleThreadedExecutor를 별도 스레드에서 스핀해 GL 스레드 일관성을 보장해야 함.
        if len(self.camera_dict["cam_names"]) > 0:
            self._cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
            self._cam_thread.start()
        
        
    def sim_loop(self):
        t0 = time.perf_counter()
        
        # 1) step physics
        with self._mj_lock:
            mujoco.mj_step(self.mj_model, self.mj_data)
        t1 = time.perf_counter()
        
        # 2) copy slices → message lists
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qfrc = self.mj_data.qfrc_applied # qfrc_applied is external force
        sens = self.mj_data.sensordata
        
        for i, (idx_q, nq, idx_v, nv, _) in enumerate(self._j_slices):
            self._pos_lists[i][:] = array('d', qpos[idx_q: idx_q + nq])
            self._vel_lists[i][:] = array('d', qvel[idx_v: idx_v + nv])
            self._tau_lists[i][:] = array('d', qfrc[idx_v: idx_v + nv])
        for i, (idx_s, ns, _) in enumerate(self._s_slices):
            self._sensor_lists[i][:] = array('d', sens[idx_s: idx_s + ns])
        t2 = time.perf_counter()

        # 3) publish bundles
        now = self.get_clock().now().to_msg()

        joint_dict_msg = JointDict()
        joint_dict_msg.header.stamp = now
        joint_dict_msg.sim_time = to_TimeMsg(self.mj_data.time)
        joint_dict_msg.positions  = self._pos_msgs
        joint_dict_msg.velocities = self._vel_msgs
        joint_dict_msg.torques    = self._tau_msgs
        self.joint_dict_pub.publish(joint_dict_msg)

        sensor_dict_msg = SensorDict()
        sensor_dict_msg.header.stamp = now
        sensor_dict_msg.sim_time = to_TimeMsg(self.mj_data.time)
        sensor_dict_msg.sensors = self._sensor_msgs
        self.sensor_dict_pub.publish(sensor_dict_msg)
        t3 = time.perf_counter()
        

        
        durations = {
            "mj_step":     (t1 - t0)*1000,
            "getData":     (t2 - t1)*1000,
            "publishData": (t3 - t2)*1000,
            "totalStep":   (t3 - t0)*1000
        }
        
        if self.dt - (time.perf_counter() - t0) < 0:
            lines = [
                "\n===================================",
                f"mj_step took {durations['mj_step']:.6f} ms",
                f"getData took {durations['getData']:.6f} ms",
                f"publishData took {durations['publishData']:.6f} ms",
                f"totalStep took {durations['totalStep']:.6f} ms",
                "===================================",
            ]
            # self.get_logger().warn("\n".join(lines))

    def viewer_loop(self):
        try:
            with self._mj_lock:
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
        except Exception:
            pass

    def camera_thread(self):
        try:
            self.cam_renderer = mujoco.Renderer(self.mj_model, width=self.camera_dict["resolution"][0], height=self.camera_dict["resolution"][1])
            self.cam_renderer.disable_depth_rendering()
        except Exception as e:
            self.get_logger().error(f"[camera_thread] failed to create renderer: {e}")
            return

        period = 1.0 / self.camera_fps
        next_t = time.perf_counter()

        try:
            while rclpy.ok():
                t0 = time.perf_counter()
                
                # 1) copy minimal state
                with self._mj_lock:
                    rd = self._render_data
                    sd = self.mj_data
                    rd.qpos[:] = sd.qpos
                    rd.qvel[:] = sd.qvel
                    if self.mj_model.nu:
                        rd.act[:] = sd.act
                    if self.mj_model.nmocap:
                        rd.mocap_pos[:]  = sd.mocap_pos
                        rd.mocap_quat[:] = sd.mocap_quat
                    rd.time = sd.time

                # 2) forward kinematics/dynamics, render
                mujoco.mj_forward(self.mj_model, self._render_data)

                stamp = self.get_clock().now().to_msg()

                # 3) 각 카메라 렌더 → 사전할당 버퍼(bytearray)에 복사
                for cname in self.camera_dict["cam_names"]:
                    try:
                        self.cam_renderer.update_scene(self._render_data, camera=cname)
                        rgb = self.cam_renderer.render()  # (H,W,3) uint8
                        self._cam_buf_views[cname][:] = rgb.reshape(-1)
                        self._cam_named[cname].image.header.stamp = stamp
                    except Exception as e:
                        self.get_logger().warn(f"[camera_thread] {cname}: {e}")

                # 3) publish image bundle
                self._imgdict_msg.header.stamp = stamp
                self._imgdict_msg.sim_time = to_TimeMsg(self._render_data.time)
                self.image_pub.publish(self._imgdict_msg)


                loop_duration = time.perf_counter() - t0
                if(loop_duration < period):
                    time.sleep(period - loop_duration)
        finally:
            try:
                self.cam_renderer = None
            except Exception:
                pass

    def sub_ctrl_cb(self, msg: CtrlDict):
        ctrl_command = {}
        for item in msg.commands:
            data = list(item.value.data)  # sequence → list
            ctrl_command[item.name] = data if len(data) > 1 else (data[0] if data else 0.0)
        with self.state_lock:
            for name, cmd in ctrl_command.items():
                if name in self.actuator_dict["actuator_names"]:
                    aid = self.actuator_dict["aname_to_aid"][name]
                    self.mj_data.ctrl[aid] = float(cmd)

    def pub_joint_cb(self):
        joint_names = self.joint_dict["joint_names"]
        if not joint_names:
            return

        positions, velocities = [], []
        for jname in joint_names:
            jid   = self.joint_dict["jname_to_jid"][jname]
            idx_q = self.mj_model.jnt_qposadr[jid]
            idx_v = self.mj_model.jnt_dofadr[jid]
            next_q = (self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq)
            next_v = (self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv)
            nq = next_q - idx_q
            nv = next_v - idx_v
            positions.extend(self.mj_data.qpos[idx_q: idx_q + nq].tolist())
            velocities.extend(self.mj_data.qvel[idx_v: idx_v + nv].tolist())

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = positions
        msg.velocity = velocities
        self.joint_state_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.cam_thread.is_alive():
                self.cam_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.close()
        except Exception:
            pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()