### 데이터
> - 직접 수집한 데이터 : <a href="https://drive.google.com/drive/folders/0AAjFYj8gqRE5Uk9PVA"> 링크 </a>
> - kaggle 데이터     : <a href="https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download"> 링크 </a>

### 라벨링
> - <a href="https://www.robots.ox.ac.uk/~vgg/software/via/via.html"> VIA Tool 이용 </a>
> - Attributes 부여하지 않고 Bbox 형태로 라벨링 진행
> - 라벨링 완료 후 export json 클릭하여 json 파일로 저장
> - 파일명을 (0_to_100.json) 형태로 바꾼뒤 Annotation_json_files에 업로드

### 라벨링 Rule
> - 번호판 종류와 상관없이 번호판 좌상단 끝부분과 우하단 끝부분을 라벨링
> - 최대한 tight하게 진행
> - 번호판 내 TexT가 인식되지 않은 번호판은 라벨링 X

### 역할
> - 범하 : 0   ~ 99  번까지
> - 지상 : 100 ~ 199 번까지
> - 기현 : 200 ~ 326 번까지
