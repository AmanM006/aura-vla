export interface SimState {
  timestamp: number;
  phase: string;
  sub_goals: {
    drawer: boolean;
    fork: boolean;
    spoon: boolean;
    plate: boolean;
    mug: boolean;
  };
  task_success: boolean;
  npu_ms: number;
  igpu_ms: number;
  cpu_ms: number;
  front_cam?: string | null;
  overhead_cam?: string | null;
  left_wrist_cam?: string | null;
  right_wrist_cam?: string | null;
  transcript: string;
  settled_text: string;
  plan: string[];
  plan_index: number;
  objects: Record<string, [number, number, number]>;
  anomaly_score: number;
  anomaly_is_defect: boolean;
  anomaly_hotspot: string;
  anomaly_device: string;
  policy_mode: string;
  temporal_ensemble_active: boolean;
}
