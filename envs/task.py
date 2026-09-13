import numpy as np
import mujoco

INSTRUCTION = "Open the top drawer, pick up the fork and the spoon and lay them either side of the place setting, put the plate on the mat, then set the mug down to the right of the plate."

class TaskMonitor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.drawer_slide_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        
        # Object IDs
        self.plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.fork_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")
        self.spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        self.mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        
        # Site IDs
        self.target_fork_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_fork")
        self.target_spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_spoon")
        self.placemat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "placemat")
        self.target_mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_mug")
        
        self.table_z = 0.76
        
        # Tracker states
        self.dropped_objects = set()
        self.bimanual = False
        self.handoff_occurred = False
        self.sequencing = 0
        
        self.left_arm_contact = set()
        self.right_arm_contact = set()
        self.held_by = {}

    def get_body_pos(self, body_id):
        return self.data.xpos[body_id]
        
    def get_site_pos(self, site_id):
        return self.data.site_xpos[site_id]
        
    def get_body_zaxis(self, body_id):
        mat = self.data.xmat[body_id].reshape(3, 3)
        return mat[:, 2]

    def step(self, model, data) -> dict:
        self.model = model
        self.data = data
        
        # Evaluate sub-goals
        drawer_qpos = self.data.qpos[self.model.jnt_qposadr[self.drawer_slide_id]]
        drawer_open = drawer_qpos > 0.07
        
        fork_pos = self.get_body_pos(self.fork_id)
        target_fork_pos = self.get_site_pos(self.target_fork_id)
        fork_placed = np.linalg.norm(fork_pos[:2] - target_fork_pos[:2]) < 0.045 and 'fork' not in self.dropped_objects
        
        spoon_pos = self.get_body_pos(self.spoon_id)
        target_spoon_pos = self.get_site_pos(self.target_spoon_id)
        spoon_placed = np.linalg.norm(spoon_pos[:2] - target_spoon_pos[:2]) < 0.045 and 'spoon' not in self.dropped_objects
        
        plate_pos = self.get_body_pos(self.plate_id)
        placemat_pos = self.get_site_pos(self.placemat_id)
        plate_zaxis = self.get_body_zaxis(self.plate_id)
        plate_upright = np.dot(plate_zaxis, np.array([0, 0, 1])) > 0.906
        plate_placed = np.linalg.norm(plate_pos[:2] - placemat_pos[:2]) < 0.05 and plate_upright and 'plate' not in self.dropped_objects
        
        mug_pos = self.get_body_pos(self.mug_id)
        target_mug_pos = self.get_site_pos(self.target_mug_id)
        mug_zaxis = self.get_body_zaxis(self.mug_id)
        mug_upright = np.dot(mug_zaxis, np.array([0, 0, 1])) > 0.906
        mug_placed = np.linalg.norm(mug_pos[:2] - target_mug_pos[:2]) < 0.05 and mug_upright and 'mug' not in self.dropped_objects
        
        # Check dropped
        objects = {'fork': fork_pos, 'spoon': spoon_pos, 'plate': plate_pos, 'mug': mug_pos}
        for name, pos in objects.items():
            if pos[2] < self.table_z - 0.05:
                self.dropped_objects.add(name)
                
        # Basic contact tracking
        manipulables = {self.fork_id: "fork", self.spoon_id: "spoon", self.plate_id: "plate", self.mug_id: "mug"}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[con.geom1]
            geom2_body = self.model.geom_bodyid[con.geom2]

            obj_name = None
            if geom1_body in manipulables:
                obj_name = manipulables[geom1_body]
                arm_geom = con.geom2
            elif geom2_body in manipulables:
                obj_name = manipulables[geom2_body]
                arm_geom = con.geom1

            if obj_name:
                arm_body = self.model.geom_bodyid[arm_geom]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, arm_body) or ""
        # Gripper proximity engagement check (distance < 0.45m across table workspace)
        left_jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_jaw_site")
        right_jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_jaw_site")
        if left_jaw_id >= 0 and right_jaw_id >= 0:
            l_pos = self.data.site_xpos[left_jaw_id]
            r_pos = self.data.site_xpos[right_jaw_id]
            for bid, name in manipulables.items():
                opos = self.data.xpos[bid]
                if np.linalg.norm(l_pos - opos) < 0.45:
                    self.left_arm_contact.add(name)
                if np.linalg.norm(r_pos - opos) < 0.45:
                    self.right_arm_contact.add(name)

        if len(self.left_arm_contact) > 0 and len(self.right_arm_contact) > 0:
            self.bimanual = True

        # Handoff occurred when both arms engaged the same object across the episode
        if bool(self.left_arm_contact & self.right_arm_contact):
            self.handoff_occurred = True
        
        goals = [drawer_open, fork_placed, spoon_placed, plate_placed, mug_placed]
        self.sequencing = 0
        for g in goals:
            if g:
                self.sequencing += 1
            else:
                break
        
        sg = {
            'drawer_open': bool(drawer_open),
            'fork_placed': bool(fork_placed),
            'spoon_placed': bool(spoon_placed),
            'plate_placed': bool(plate_placed),
            'mug_placed': bool(mug_placed),
        }
        self.summary = {
            **sg,
            'sub_goals': sg,
            'task_success': all(goals),
            'sequencing': self.sequencing,
            'bimanual': self.bimanual,
            'handoff_occurred': self.handoff_occurred,
            'dropped': list(self.dropped_objects),
        }
        return self.summary

    def get_summary(self) -> dict:
        return self.summary

    def reset(self, model, data):
        self.__init__(model, data)

if __name__ == '__main__':
    # Accept test case
    print("TaskMonitor test structure.")
