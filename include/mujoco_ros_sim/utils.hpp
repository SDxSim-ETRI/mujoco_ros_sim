#include <mujoco/mujoco.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>
namespace MujocoRosSim
{

  static inline const char* name_or(const mjModel* m, int objtype, int id) {
    const char* n = mj_id2name(m, objtype, id);
    return n ? n : "-";
  }
  
  static std::string JointTypeName(int t) {
    switch (t) {
      case mjJNT_FREE:  return "Free";
      case mjJNT_BALL:  return "Ball";
      case mjJNT_SLIDE: return "Slide";
      case mjJNT_HINGE: return "Hinge";
      default:          return "Unk";
    }
  }
  
  static std::string TrnTypeName(int t) {
    switch (t) {
      case mjTRN_JOINT:           return "Joint";
      case mjTRN_JOINTINPARENT:   return "JointInParent";
      case mjTRN_SLIDERCRANK:     return "SliderCrank";
      case mjTRN_TENDON:          return "Tendon";
      case mjTRN_SITE:            return "Site";
      case mjTRN_BODY:            return "Body";
      default:                    return "Unk";
    }
  }
  
  static std::string SensorTypeName(int t) {
    switch (t) {
      case mjSENS_ACCELEROMETER:  return "Accelerometer";
      case mjSENS_VELOCIMETER:    return "Velocimeter";
      case mjSENS_GYRO:           return "Gyro";
      case mjSENS_FORCE:          return "Force";
      case mjSENS_TORQUE:         return "Torque";
      case mjSENS_MAGNETOMETER:   return "Magnetometer";
      case mjSENS_RANGEFINDER:    return "Rangefinder";
      case mjSENS_JOINTPOS:       return "JointPos";
      case mjSENS_JOINTVEL:       return "JointVel";
      case mjSENS_TENDONPOS:      return "TendonPos";
      case mjSENS_TENDONVEL:      return "TendonVel";
      case mjSENS_ACTUATORFRC:    return "ActuatorFrc";
      case mjSENS_ACTUATORVEL:    return "ActuatorVel";
      case mjSENS_ACTUATORPOS:    return "ActuatorPos";
      case mjSENS_BALLQUAT:       return "BallQuat";
      case mjSENS_SUBTREECOM:     return "SubtreeCom";
      case mjSENS_SUBTREELINVEL:  return "SubtreeLinVel";
      case mjSENS_SUBTREEANGMOM:  return "SubtreeAngMom";
      case mjSENS_FRAMEPOS:       return "FramePos";
      case mjSENS_FRAMEQUAT:      return "FrameQuat";
      case mjSENS_FRAMEXAXIS:     return "FrameXAxis";
      case mjSENS_FRAMEYAXIS:     return "FrameYAxis";
      case mjSENS_FRAMEZAXIS:     return "FrameZAxis";
      case mjSENS_FRAMELINVEL:    return "FrameLinVel";
      case mjSENS_FRAMEANGVEL:    return "FrameAngVel";
      case mjSENS_FRAMELINACC:    return "FrameLinAcc";
      case mjSENS_FRAMEANGACC:    return "FrameAngAcc";
      default:                    return "Unk";
    }
  }
  
  static std::string ObjTypeName(int t) {
    switch (t) {
      case mjOBJ_BODY:      return "Body";
      case mjOBJ_JOINT:     return "Joint";
      case mjOBJ_GEOM:      return "Geom";
      case mjOBJ_SITE:      return "Site";
      case mjOBJ_CAMERA:    return "Camera";
      case mjOBJ_LIGHT:     return "Light";
      case mjOBJ_TENDON:    return "Tendon";
      case mjOBJ_ACTUATOR:  return "Actuator";
      case mjOBJ_SENSOR:    return "Sensor";
      default:              return "-";
    }
  }
  
  // ---- 메인 함수 ---------------------------------------------------------------
  
  std::string PrintTable(const std::string& robot_name, const mjModel* m) {
    if (!m) return "null model\n";
  
    std::ostringstream out;
    out << "\n=================================================================\n";
    out << "=================================================================\n";
    out << "MuJoCo Model Information: " << robot_name << "\n";
    out << " id | name                 | type   | nq | nv | idx_q | idx_v\n";
    out << "----+----------------------+--------+----+----+-------+------\n";
  
    // Joints
    for (int jid = 0; jid < m->njnt; ++jid) {
      const char* name = name_or(m, mjOBJ_JOINT, jid);
      if (!name || !*name) continue;
  
      int jtype = m->jnt_type[jid];
      int idx_q = m->jnt_qposadr[jid];
      int idx_v = m->jnt_dofadr[jid];
      int next_q = (jid + 1 < m->njnt) ? m->jnt_qposadr[jid + 1] : m->nq;
      int next_v = (jid + 1 < m->njnt) ? m->jnt_dofadr[jid + 1] : m->nv;
      int nq = next_q - idx_q;
      int nv = next_v - idx_v;
  
      out << std::setw(3) << jid << " | "
          << std::left << std::setw(20) << name << " | "
          << std::left << std::setw(6) << JointTypeName(jtype) << " | "
          << std::right << std::setw(2) << nq << " | "
          << std::right << std::setw(2) << nv << " | "
          << std::right << std::setw(5) << idx_q << " | "
          << std::right << std::setw(4) << idx_v << "\n";
    }
  
    // Actuators
    out << "\n";
    out << " id | name                 | trn     | target_joint\n";
    out << "----+----------------------+---------+-------------\n";
    for (int aid = 0; aid < m->nu; ++aid) {
      const char* aname = name_or(m, mjOBJ_ACTUATOR, aid);
      int trn_type = m->actuator_trntype[aid];

    int jid = -1;
    #ifdef mjNTRN
    for (int k = 0; k < mjNTRN; ++k) {
        int cand = m->actuator_trnid[aid * mjNTRN + k];
        if (cand >= 0) { jid = cand; break; }
    }
    #else
    for (int k = 0; k < 2; ++k) {
        int cand = m->actuator_trnid[aid * 2 + k];
        if (cand >= 0) { jid = cand; break; }
    }
    #endif

    std::string target_joint = "-";
    if (jid >= 0 && jid < m->njnt)
    target_joint = name_or(m, mjOBJ_JOINT, jid);
    else if (jid >= 0)
    target_joint = std::to_string(jid);

  
      out << std::right << std::setw(3) << aid << " | "
          << std::left << std::setw(20) << (aname ? aname : "-") << " | "
          << std::left << std::setw(7)  << TrnTypeName(trn_type) << " | "
          << target_joint << "\n";
    }
  
    // Sensors
    out << "\n";
    out << " id | name                        | type             | dim | adr | target (obj)\n";
    out << "----+-----------------------------+------------------+-----+-----+----------------\n";
    for (int sid = 0; sid < m->nsensor; ++sid) {
      const char* sname = name_or(m, mjOBJ_SENSOR, sid);
      int stype   = m->sensor_type[sid];
      int dim     = m->sensor_dim[sid];
      int sadr    = m->sensor_adr[sid];
      int objtype = m->sensor_objtype[sid];
      int objid   = m->sensor_objid[sid];
  
      std::string target("-");
      if (objid >= 0) {
        std::string otype = ObjTypeName(objtype);
        const char* oname = nullptr;
        if      (objtype == mjOBJ_BODY)  oname = name_or(m, mjOBJ_BODY,  objid);
        else if (objtype == mjOBJ_SITE)  oname = name_or(m, mjOBJ_SITE,  objid);
        else if (objtype == mjOBJ_JOINT) oname = name_or(m, mjOBJ_JOINT, objid);
        target = otype + ":" + (oname ? oname : std::to_string(objid));
      }
  
      out << std::right << std::setw(3) << sid << " | "
          << std::left  << std::setw(27) << (sname ? sname : "-") << " | "
          << std::left  << std::setw(16) << SensorTypeName(stype) << " | "
          << std::right << std::setw(3)  << dim << " | "
          << std::right << std::setw(3)  << sadr << " | "
          << target << "\n";
    }
  
    // Cameras
    out << "\n";
    out << " id | name                        | mode     | resolution\n";
    out << "----+-----------------------------+----------+------------\n";
  
    int offw = (m->vis.global.offwidth  > 0) ? m->vis.global.offwidth  : 0;
    int offh = (m->vis.global.offheight > 0) ? m->vis.global.offheight : 0;
    std::string res_str = (offw > 0 && offh > 0) ? (std::to_string(offw) + "x" + std::to_string(offh)) : "-";
  
    for (int cid = 0; cid < m->ncam; ++cid) {
      const char* cname = name_or(m, mjOBJ_CAMERA, cid);
  
      // cam mode: 공개 필드가 없는 빌드가 많아 안전하게 "-"
      std::string mode_str = "-";
      // (필요하면 프로젝트에서 별도 확장 저장해 쓰세요)
  
      out << std::right << std::setw(3) << cid << " | "
          << std::left  << std::setw(27) << (cname ? cname : "-") << " | "
          << std::left  << std::setw(8)  << mode_str << " | "
          << res_str << "\n";
    }
  
    out << "=================================================================\n";
    out << "=================================================================\n";
    return out.str();
  }
} // namespace MujocoRosSim
