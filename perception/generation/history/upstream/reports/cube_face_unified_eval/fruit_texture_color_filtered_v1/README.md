# Fruit texture color filter v1

## 목적

기존 fruit texture root에서 과일별 색공간 outlier를 조금 줄였다. 기존 root는 수정하지 않고, 새 texture root를 만든다.

## 입력/출력

- Source: `datasets/fruit_textures/final_fruits36070_original30`
- Filtered: `datasets/fruit_textures/final_fruits36070_original30_color_filtered_v1`
- Report: `reports/cube_face_unified_eval/fruit_texture_color_filtered_v1`

## 결과

| class | before | after | removed |
| --- | ---: | ---: | ---: |
| apple | 1600 | 1470 | 130 |
| orange | 1600 | 1579 | 21 |
| banana | 1600 | 1488 | 112 |
| pineapple | 1600 | 1521 | 79 |

## 의도

- apple: red/red-orange 중심은 유지하고, yellow/green/cool outlier를 제거한다.
- orange: warm orange/red-orange/yellow 중심은 유지하고, green/cool outlier를 제거한다.
- banana: yellow banana 중심으로 유지하고, green/cool outlier를 제거한다.
- pineapple: brown/green whole pineapple 쪽은 유지하고, cool/illustration/포장 배경에 치우친 outlier를 제거한다.

색 필터는 semantic filter가 아니다. orange slice, logo, 사람, 포장 등은 hue만으로 완전히 제거되지 않을 수 있으므로 다음 단계에서 별도 semantic review가 필요하다.

## 재생성 명령

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
$py = 'C:\Users\user\anaconda3\envs\ai_robotics\python.exe'

& $py scripts\filter_fruit_textures_by_hue.py `
  --source_root datasets\fruit_textures\final_fruits36070_original30 `
  --output_root datasets\fruit_textures\final_fruits36070_original30_color_filtered_v1 `
  --report_root reports\cube_face_unified_eval\fruit_texture_color_filtered_v1 `
  --mode hardlink `
  --reset
```

## 다음 사용 예

다음 synthetic dataset 생성 또는 booster 생성에서 기존 texture root 대신 아래를 사용한다.

```powershell
-TextureRoots "datasets\fruit_textures\final_fruits36070_original30_color_filtered_v1"
```

기존 dataset이나 모델은 교체하지 않는다. 새 texture root를 사용하는 materialized dataset을 별도 이름으로 만든 뒤, 기존 model 대비 hard-case replay와 validation을 비교한다.
