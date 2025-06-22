import cv2
import numpy as np
import os
import time
import json
from glob import glob

REF_ROOT = "C:/Users/82109/OneDrive/문서/img_f/images/reference"
CUR_FOLDER = "C:/Users/82109/OneDrive/문서/img_f/images/current"
RESULT_FOLDER = "C:/Users/82109/OneDrive/문서/img_f/images/result"
JSON_PATH = os.path.join(RESULT_FOLDER, "status.json")

WEIGHTPOINT_CLASSES = {
    "01": "cola",
    "02": "doll",
    "03": "cola",
    "04": "timer",
    "05": "lemon",
    "06": "lemon",
    "07": "lemon",
    "08": "lemon",
    "09": "doll",
    "10": "clock"
}

ORB_MATCH_COUNT_THRESH = 300
EDGE_CHANGE_THRESH = 0.05
MIN_CONTOUR_AREA = 300

status_log = {}

def resize_to_fit(img, max_dim=800):
    if img is None: return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def orb_match_count(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    return len(matches)

def align_images(ref, cur):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None: return cur
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    if len(matches) < 10: return cur
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:30]]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:30]]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(cur, H, (ref.shape[1], ref.shape[0]))

def draw_largest_contour(mask, img):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    if not contours: return img, False
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    return img, True

def detect_change_by_edges(ref, cur):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    edge1 = cv2.Canny(ref_gray, 60, 130)
    edge2 = cv2.Canny(cur_gray, 60, 130)
    diff = cv2.absdiff(edge1, edge2)
    return diff

def find_best_match(reference_dir, current_img):
    best_score = -1
    best_img = None
    for path in glob(os.path.join(reference_dir, "*.jpg")):
        ref_img = resize_to_fit(cv2.imread(path))
        if ref_img is None: continue
        score = orb_match_count(ref_img, current_img)
        if score > best_score:
            best_score = score
            best_img = ref_img
    return best_img

def analyze(idx_str, current_img):
    ref_dir = os.path.join(REF_ROOT, f"reference_{idx_str}")
    best_ref_img = find_best_match(ref_dir, current_img)
    if best_ref_img is None:
        print(f"[{idx_str}] reference 이미지 없음 또는 매칭 실패")
        return current_img, False

    aligned = align_images(best_ref_img, current_img)
    diff_mask = detect_change_by_edges(best_ref_img, aligned)
    edge_ratio = np.count_nonzero(diff_mask) / diff_mask.size
    result_img = aligned.copy()
    abnormal = edge_ratio > EDGE_CHANGE_THRESH
    if abnormal:
        result_img, _ = draw_largest_contour(diff_mask, result_img)
    return result_img, abnormal

def monitor_loop():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    processed = set()
    print("[시작] current 이미지 감시 중...")
    while True:
        files = sorted(f for f in os.listdir(CUR_FOLDER) if f.endswith(".jpg"))
        new_files = [f for f in files if f not in processed]

        for file in new_files:
            idx = os.path.splitext(file)[0].zfill(2)
            current_path = os.path.join(CUR_FOLDER, file)
            result_path = os.path.join(RESULT_FOLDER, file)
            cur_img = resize_to_fit(cv2.imread(current_path))
            if cur_img is None:
                print(f"[{idx}] current 이미지 로딩 실패")
                continue

            result_img, abnormal = analyze(idx, cur_img)
            if abnormal:
                cv2.imwrite(result_path, result_img)
                status_log[idx] = "비정상"
                print(f"[{idx}] 이상 감지 → 저장 완료")
            else:
                status_log[idx] = "정상"
                print(f"[{idx}] 정상")

            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(status_log, f, ensure_ascii=False, indent=2)

            processed.add(file)
        time.sleep(5)

if __name__ == "__main__":
    monitor_loop()
