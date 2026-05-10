# 13. Phase 4 — 포트 검출 모델 (Vision-First 정렬)

> **목표**: 3 wrist camera 입력에서 SFP/SC 포트의 (a) 클래스 (b) 픽셀 bbox (c) 베이스 좌표계 3D 진입점을 추출. Stage A (coarse approach)의 시각 정렬 모듈.
>
> **기간**: 1~1.5일 (vast.ai 4090 1대)
>
> **결과물**: `aic_work/models/port_detector_v1/` — fine-tuned YOLOv8m 또는 RT-DETR.

---

## 1. 왜 별도 모듈?

ACT(Phase 3)는 end-to-end로 픽셀 → action을 학습하지만, 다음의 한계가 있다:
- **타겟 포트 명시성 부족**: NIC 카드 5개 중 정확히 어느 포트인지 task index로만 알려주면 학습 어려움
- **충돌(-24) 회피**: 정확한 포트 진입점을 알면 safe bbox clamp가 명시적으로 가능
- **새로운 카드 위치 일반화**: detector는 픽셀 기준이라 transfer 우수

→ Stage A에서 detector로 진입점을 잡고 → Stage B에서 ACT가 미세 정렬 → Stage C에서 force-guided 삽입.

---

## 2. 아키텍처 선택

| 모델 | mAP@0.5 (synthetic est.) | Latency on L4 24GB | 메모리 | 비고 |
|---|---|---|---|---|
| YOLOv8s | 0.92 | 8ms | 1GB | 가벼움, 정확도 한계 |
| **YOLOv8m** | **0.96** | **15ms** | **2GB** | **균형 — 1차 선택** |
| YOLOv8l | 0.97 | 25ms | 4GB | 약간 나음 |
| RT-DETR-r18 | 0.96 | 18ms | 2.5GB | 더 robust, 학습 느림 |
| RT-DETR-r50 | 0.97 | 35ms | 4GB | 정확하지만 latency |

**1차 선택**: YOLOv8m. ablation에서 RT-DETR-r18 비교.

---

## 3. 클래스 정의

| ID | 클래스 | 비고 |
|---|---|---|
| 0 | sfp_port | NIC card 위 8개 SFP 포트 (port_0 ~ port_7) |
| 1 | sc_port  | SC mount 위 SC 포트 |
| 2 | sfp_plug | TCP에 잡힌 SFP plug (그립 정렬에 사용) |
| 3 | sc_plug  | TCP에 잡힌 SC plug |
| 4 | nic_card | NIC card 본체 (occlusion 기준점) |
| 5 | enclosure_edge | 인클로저 모서리 (충돌 회피용) |

**중요**: detector는 단지 bbox만 — 어느 포트가 "타겟"인지는 별도 로직(task message + position)으로 결정. 같은 NIC card 안의 sfp_port 0/1 구분은 픽셀 가까운 두 bbox 중 task가 가리키는 쪽을 선택.

---

## 4. 합성 데이터 자동 라벨링

수작업 라벨링은 비효율 — Gazebo의 ground_truth TF로 모든 포트 위치를 알 수 있으므로 자동 라벨 가능.

### 4.1 자동 라벨 파이프라인

```python
# aic_work/data/auto_label_ports.py
"""
Gazebo 시뮬레이션을 띄우고 random scene generation으로 매번 다른 layout 생성.
TF에서 각 포트의 3D 위치 → 카메라 intrinsics + extrinsics로 픽셀 투영.
픽셀 (cx,cy) 주변 24×16 박스를 bbox로 라벨링 (포트 입구 크기 추정).
"""
import json
import numpy as np
import rclpy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

PORTS_TO_DETECT = [
    "nic_card_{}/sfp_port_0", "nic_card_{}/sfp_port_1", ...  # 5 cards
    "sc_mount_{}/sc_port_0", "sc_mount_{}/sc_port_1", ...
]

def project_to_pixel(p_world, K, T_cam_world):
    p_cam = T_cam_world @ np.append(p_world, 1.0)
    if p_cam[2] <= 0: return None  # behind camera
    u = K[0,0] * p_cam[0]/p_cam[2] + K[0,2]
    v = K[1,1] * p_cam[1]/p_cam[2] + K[1,2]
    return int(u), int(v), p_cam[2]
```

### 4.2 데이터 양

- 매 scene → 3 cam = 3 image (각 1152×1024)
- 5000 scene 목표 → 15,000 images
- train/val/test = 80/10/10
- 1 scene 당 평균 라벨 박스 수: 5~10 (sfp_port) + 1~2 (sc_port) + 1~3 (nic_card)

### 4.3 augmentation (학습 시, on-the-fly)

```python
# YOLOv8 hyp.yaml
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 5
translate: 0.1
scale: 0.5
shear: 2
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3   # 같은 클래스 cut-paste — port 다양성 ↑
```

**커스텀 augmentation** (코드로 추가):
- random JPEG quality 70~95
- gaussian + salt/pepper noise (camera sim에서도 적용했지만 보강)
- random lighting (HSV V scale)
- random NIC card에 occluder (다른 card / 케이블 / 손)

---

## 5. 학습 설정

```yaml
# aic_work/training/yolov8_port.yaml
path: /workspace/aic_work/datasets/port_detection
train: images/train
val:   images/val
test:  images/test
nc: 6
names: [sfp_port, sc_port, sfp_plug, sc_plug, nic_card, enclosure_edge]

# hyp
imgsz: 1024            # 1152x1024 → 조금 작게
batch: 32              # 4090 24GB
epochs: 100
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3
patience: 20
amp: true
device: 0
workers: 8
```

### 5.1 학습 명령

```bash
cd /workspace/aic_work
pixi run yolo train \
    model=yolov8m.pt \
    data=training/yolov8_port.yaml \
    imgsz=1024 epochs=100 batch=32 amp=true \
    project=models/port_detector_v1 name=run0
```

---

## 6. 평가 지표

| 지표 | 목표 (synthetic test) | 목표 (Gazebo eval scene) |
|---|---|---|
| mAP@0.5 (sfp_port) | ≥ 0.97 | ≥ 0.90 |
| mAP@0.5 (sc_port)  | ≥ 0.95 | ≥ 0.88 |
| mAP@0.5:0.95       | ≥ 0.65 | ≥ 0.55 |
| 추론 latency (1 img on L4) | ≤ 15ms | ≤ 15ms |
| 3D back-project error (median) | ≤ 1mm | ≤ 3mm |

평가 스크립트:
```bash
pixi run yolo val \
    model=models/port_detector_v1/run0/weights/best.pt \
    data=training/yolov8_port.yaml \
    imgsz=1024 device=0
```

---

## 7. 3D back-projection

bbox 중심 (u,v)을 카메라 좌표계 ray로 → port plane intersection.

전제: NIC card / SC mount는 task_board 평면에 부착 → 평면 방정식을 알면 ray-plane intersection으로 3D 좌표.

```python
def pixel_to_3d_on_plane(uv, K, T_world_cam, plane_normal_world, plane_point_world):
    # 1. 픽셀 → 카메라 좌표 ray
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    ray_cam = np.array([(uv[0]-cx)/fx, (uv[1]-cy)/fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    # 2. world ray
    R_world_cam = T_world_cam[:3,:3]; t_world_cam = T_world_cam[:3,3]
    ray_world = R_world_cam @ ray_cam
    # 3. plane intersection
    denom = plane_normal_world @ ray_world
    if abs(denom) < 1e-6: return None
    t = (plane_normal_world @ (plane_point_world - t_world_cam)) / denom
    return t_world_cam + t * ray_world
```

평면 정보:
- task_board의 mount surface는 SDF에서 알 수 있고, 첫 frame TF lookup으로 동적으로 계산 가능 (ground_truth=true 시)
- **평가 시에는 ground_truth 사용 불가** → 다음 중 하나:
  1. 첫 몇 프레임 동안 nic_card bbox 평면 fitting (PCA)
  2. depth가 없는 RGB-only이므로 stereo baseline 활용 (left/right cam 매칭)
  3. 사전 정의된 task_board 좌표에 회귀

**선택**: stereo baseline triangulation. left/right 카메라가 모두 보고 있는 동일 포트는 disparity로 3D 계산.

```python
def triangulate(uv_left, uv_right, K_l, K_r, T_l, T_r):
    # OpenCV cv2.triangulatePoints
    P_l = K_l @ np.linalg.inv(T_l)[:3]
    P_r = K_r @ np.linalg.inv(T_r)[:3]
    pts4d = cv2.triangulatePoints(P_l, P_r, uv_left, uv_right)
    return (pts4d[:3] / pts4d[3]).flatten()
```

---

## 8. detection 모듈 인터페이스

`aic_work/policies/port_detector.py`:

```python
class PortDetector:
    def __init__(self, ckpt_path, device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(ckpt_path)
        self.device = device

    def detect(self, image_left, image_center, image_right):
        """
        Returns: dict with detections per camera
            {
              "left":  [{"cls":0, "bbox":(x1,y1,x2,y2), "conf":0.97}, ...],
              "center":[...], "right":[...]
            }
        """
        results = self.model.predict(
            [image_left, image_center, image_right],
            imgsz=1024, conf=0.4, iou=0.5, device=self.device, verbose=False
        )
        return self._parse(results)

    def get_target_port_3d(self, dets, task, K_left, T_left, K_right, T_right):
        """
        task: 'sfp' or 'sc' + target_module_name
        Returns: 3D entry point in base_link, or None.
        """
        # 1. left/right에서 task와 일치하는 cls의 bbox 매칭
        # 2. epipolar line 검증
        # 3. triangulate
        # 4. base_link로 변환
        ...
```

---

## 9. 실패 모드 + 방어

| 실패 모드 | 방어 |
|---|---|
| occlusion으로 left/right 한쪽만 detection | 단안 + 평면 prior 활용 (NIC card bbox로 평면 fitting) |
| 같은 NIC에 sfp_port_0/1 모두 검출 → 어느 게 타겟? | task message 기반: 타겟 module의 첫 hit ports를 task index로 매핑 |
| confidence 낮음 (< 0.4) | 카메라 noise 의심 → frame averaging (5 frame 평균) |
| 3D 좌표 ±5mm 이상 오차 | Stage B (ACT)가 처리 — Stage A는 ±1cm로 충분 |
| 모든 port detection 실패 | fallback: task_board pose에 prior offset 적용 (덜 정확하지만 동작) |

---

## 10. 합성 데이터 한계 → real-eval gap

Gazebo의 합성 이미지로 학습한 detector가 실제 Gazebo eval 환경 (다른 lighting, texture)에서 잘 작동할지 확인. mAP 차이가 크면:
- Phase 6에서 Isaac/MuJoCo 렌더로 추가 학습
- 또는 SFP/SC plug의 텍스처를 더 다양하게 randomize

---

## 11. 결정 로그

- **결정**: YOLOv8m (1차) — RT-DETR ablation은 Phase 7
- **결정**: 합성 5000 scene × 3 cam = 15K images
- **결정**: stereo triangulation으로 3D 추정 (평가 시 ground_truth 불가)
- **결정**: 클래스 6개 — enclosure_edge는 충돌 회피 사이드 데이터로 추가

---

## 12. 완료 기준

- [ ] 합성 데이터 15K images (자동 라벨)
- [ ] YOLOv8m fine-tune 완료, mAP@0.5 ≥ 0.95 (synthetic test)
- [ ] Gazebo eval scene 100장에서 mAP@0.5 ≥ 0.85
- [ ] 추론 latency ≤ 15ms (L4)
- [ ] PortDetector 클래스 → ACT의 Stage A에서 동작 확인
