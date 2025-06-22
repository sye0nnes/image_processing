# image_processing

 이상 탐지 시스템 구현을 위한 코드입니다.
##  폴더 구조
- `images/reference/` – 기준 이미지
- `images/current/` – 현재 촬영된 이미지
- `images/result/` – 분석 결과 저장

##  주요 기능
- ORB 기반 이미지 정렬 및 유사도 비교
- 중심부 강조 필터 적용
- 이상 영역 바운딩 박스 시각화
- 5초 간격 자동 이미지 감시 루프
- 이상 여부를 JSON 상태

##  실행 방법

```bash
python object_anomaly_detector
