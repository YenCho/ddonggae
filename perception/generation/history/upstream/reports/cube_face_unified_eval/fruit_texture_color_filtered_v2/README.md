# fruit texture color filtered v2

목적: `final_fruits36070_original30`에서 과일별 색상 outlier를 조금 더 줄여서, apple이 orange/yellow 색공간을 과하게 침범하는 현상을 완화한다.

## 결과

원본 texture root는 수정하지 않았고, 새 후보 root만 만들었다.

- source: `datasets/fruit_textures/final_fruits36070_original30`
- output: `datasets/fruit_textures/final_fruits36070_original30_color_filtered_v2`
- report: `reports/cube_face_unified_eval/fruit_texture_color_filtered_v2`

클래스별 1600장 중 160장씩 제거해서 10% reject, 1440장 keep으로 맞췄다.

| class | before | kept v2 | rejected |
| --- | ---: | ---: | ---: |
| apple | 1600 | 1440 | 160 |
| orange | 1600 | 1440 | 160 |
| banana | 1600 | 1440 | 160 |
| pineapple | 1600 | 1440 | 160 |

## v1 대비 변경점

v1은 hard threshold만 적용해서 apple 8.1%, orange 1.3%, banana 7.0%, pineapple 4.9% 정도만 제거했다. v2는 hard threshold를 먼저 적용한 뒤, 부족한 만큼 클래스별 soft outlier score로 채워 각 클래스 10% 제거율을 맞춘다.

apple은 추가 제거분 30장을 모두 `apple_target10_orange_yellow_soft`로 뽑았다. 즉 apple texture 중 orange/yellow 계열로 밀리는 샘플을 v1보다 더 줄였다.

dominant hue bucket 기준 변화:

| class | split | red_wrap | red_orange | orange | yellow | yellow_green | green | cool |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apple | before | 1119 | 278 | 166 | 13 | 20 | 0 | 4 |
| apple | kept_v2 | 1070 | 257 | 109 | 4 | 0 | 0 | 0 |
| orange | before | 36 | 812 | 752 | 0 | 0 | 0 | 0 |
| orange | kept_v2 | 9 | 727 | 704 | 0 | 0 | 0 | 0 |
| banana | before | 4 | 17 | 1424 | 87 | 40 | 2 | 26 |
| banana | kept_v2 | 0 | 9 | 1360 | 71 | 0 | 0 | 0 |
| pineapple | before | 5 | 984 | 481 | 54 | 40 | 1 | 35 |
| pineapple | kept_v2 | 0 | 954 | 421 | 29 | 36 | 0 | 0 |

## 확인한 preview

- `rejected_apple_soft_target_contact_sheet.jpg`: apple 추가 제거분만 모은 sheet. 노란 사과, 연한 사과 단면처럼 보이는 crop, orange/yellow 계열로 밀린 샘플이 주로 제거됐다.
- `rejected_orange_soft_target_contact_sheet.jpg`: orange의 hue-edge/cool/green soft outlier 제거분.
- `before_after_dominant_h_histograms.png`: class별 dominant H 분포 before/after.

## 재생성 명령

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
$py = 'C:\Users\user\anaconda3\envs\ai_robotics\python.exe'
& $py scripts\filter_fruit_textures_by_hue.py `
  --source_root datasets\fruit_textures\final_fruits36070_original30 `
  --output_root datasets\fruit_textures\final_fruits36070_original30_color_filtered_v2 `
  --report_root reports\cube_face_unified_eval\fruit_texture_color_filtered_v2 `
  --mode hardlink `
  --target_reject_ratio 0.10 `
  --reset
```

## 주의

이 필터는 색상 기반이다. orange slice, 포장, 이상한 crop 같은 semantic outlier를 완전히 제거하는 용도는 아니다. 그래도 apple의 orange/yellow 침범을 줄이는 목적에는 v1보다 더 적합하다.
