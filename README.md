# IF 60 Studio

**"만약 ~라면?" 상상 쇼츠를 3D 복셀 애니메이션으로 만드는 제작 도구** (v0.3)

장면 구성 YAML 하나로 9:16 · 1080×1920 · 30fps 쇼츠를 렌더링합니다.
블록으로 지은 3D 도시에 수천 명의 군중을 세우고, 시계 HUD에 맞춰 새벽→낮→밤으로 하늘이 바뀌며, 2~5단어씩 끊어 읽는 자막이 올라갑니다.
AI 영상 생성 크레딧 없이 컴퓨터에서 직접 렌더링하므로 편당 추가 비용이 거의 없습니다.

![3D 복셀 예제](docs/voxel_scenes.png)
*예제 `examples/seoul_moon_gravity.yaml` — "서울 중력이 1시간 동안 달처럼 약해진다면?" (1분 51초, 18장면)*

---

## ⚡ 가장 쉬운 사용법: 주제만 입력하기

1. 저장소 폴더에서 **`IF60 만들기.command`** 를 더블클릭 (Windows는 `IF60 만들기.bat`)
2. 터미널 창에 주제를 입력하고 Enter — 예: `파리는 어떻게 그렇게 잘 도망칠까?`
3. 기다리면 끝: **웹 조사 → 대본·연출 작성 → 렌더링 → 영상 자동 재생**
   (조사·작성 1~3분 + 렌더링 수 분)

- 처음 한 번은 Anthropic API 키를 물어봅니다. https://console.anthropic.com → API Keys → Create Key로 만든 키를 붙여넣으면
  이 컴퓨터의 `.env` 파일에만 저장되고 GitHub에는 올라가지 않아요. (바꾸려면 `python -m if60 key`)
- 비용: 편당 조사·대본 작성에 대략 수백 원 수준 (Sonnet 5.5 + 웹 검색 기준 추정). 렌더링은 무료.
- 완성되면 터미널에 **확인할 사실 목록과 출처**가 나옵니다. 올리기 전에 꼭 확인하세요.
- Mac에서 "확인되지 않은 개발자" 경고가 뜨면: 파일을 우클릭 → 열기 → 열기.

터미널에서 직접 하려면:
```bash
python -m if60 make "파리는 어떻게 그렇게 잘 도망칠까?"            # 1080×1920 완성본
python -m if60 make "파리는 어떻게 그렇게 잘 도망칠까?" --preview  # 빠른 저화질 확인
python -m if60 make "주제" --length 90                               # 길이 지정
```

### 만든 영상은 어디에?
저장소 폴더 안의 **`out`** 폴더에 쌓입니다. 장면 파일(대본)은 **`scenes`** 폴더에 저장돼요.
```bash
open out        # Mac: Finder로 out 폴더 열기 (Windows: explorer out)
```

---

## 1. 설치 (처음 한 번)

**필요한 것:** Python 3.10 이상, ffmpeg

### macOS
```bash
brew install python ffmpeg          # Homebrew가 없으면 https://brew.sh 먼저 설치
git clone https://github.com/yongjujun/if60-studio.git
cd if60-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
git clone https://github.com/yongjujun/if60-studio.git
cd if60-studio
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
설치 후 PowerShell을 새로 열어야 `ffmpeg` 명령이 인식됩니다.

### 설치 확인
```bash
python -m if60 styles     # 스타일 목록이 나오면 성공
```

> 비공개 저장소라 `git clone` 할 때 GitHub 로그인을 요구할 수 있어요. 브라우저 로그인 창이 뜨면 로그인하면 됩니다.

---

## 2. 영상 만들기

`.venv`를 켠 상태(`source .venv/bin/activate` 또는 `.venv\Scripts\activate`)에서 실행합니다.

```bash
# ① 빠른 미리보기 (360×640, 15fps) — 구성과 타이밍 확인용
python -m if60 render examples/seoul_moon_gravity.yaml --preview

# ② 최종본 (1080×1920, 30fps) — 업로드용
python -m if60 render examples/seoul_moon_gravity.yaml

# 장면별 대표 프레임을 PNG로도 저장 (썸네일 고를 때)
python -m if60 render examples/seoul_moon_gravity.yaml --stills out/stills

# 길이 맞추기 (모든 장면 길이를 비례 조정)
python -m if60 render examples/seoul_moon_gravity.yaml --length 60
```
결과물은 `out/` 폴더에 저장됩니다.

**렌더링 시간 참고** (1분 51초 예제 기준)
| 환경 | 미리보기 | 최종본 |
|---|---|---|
| GPU 없는 서버 (CPU 소프트웨어 렌더링) | 약 5분 | 약 30분 |
| 일반 노트북/데스크톱 GPU (Apple Silicon, 내장/외장 그래픽) | 1분 내외 | 수 분 (예상) |

---

## 3. 새 회차 만들기

1. `examples/seoul_moon_gravity.yaml`을 복사해 `scenes/새주제.yaml`로 저장
2. 장면마다 `caption`(대본), `world`(장소), `shot`(카메라), `crowds`(군중)를 고치기
3. `--preview`로 확인 → 최종본 렌더링

### 장면 하나의 구조
```yaml
- duration: 6                           # 초
  caption: 새벽 5시 30분 광화문 광장. 중력이 6분의 1로 줄었어요.   # 2~5단어씩 자동으로 끊겨 나옴
  title: 서울 중력이 1시간 달만큼 약해진다면?                       # (선택) 상단 제목 상자
  hud: {from: "05:32", to: "05:35", step: 1}   # 우측 상단 시계. 하늘·조명도 이 시각을 따라감
  world: {template: plaza, seed: 3}           # 장소 템플릿
  shot: forward                               # 템플릿이 제공하는 카메라 샷
  crowds:                                     # 군중 (여러 개 가능)
    - {agent: person, count: 1800, area: plaza, facing: 0, bob: 0.9, bob_freq: 1.8}
    - {agent: cat, count: 300, area: plaza, bob: 1.2, scale: [1.6, 2.0]}
  effects:                                    # (선택) 연출 효과
    - shake: {at: 1.0, length: 1.4, amp: 5}
```

### 장소 템플릿과 샷
| 템플릿 | 장소 | 샷 (`shot:`) | 군중 구역 (`area:`) |
|---|---|---|---|
| `plaza` | 넓은 광장 + 도로 + 빌딩 + 성문 + 뒷산 | `forward` `aerial` `low` `rise` `gate` | `plaza` `road_l` `road_r` `sidewalk_l` `sidewalk_r` |
| `boulevard` | 고층 빌딩 협곡 사이 대로 | `forward` `sidewalk` `canyon` `up` | `road` `sidewalk_l` `sidewalk_r` |
| `crossroad` | 위에서 본 교차로 + 횡단보도 | `top` `top_drift` `street` | `lane_s` `lane_n` `lane_e` `lane_w` `crosswalk_n/s/e/w` `corner_ne/nw/se/sw` |
| `river_bridge` | 강 + 트러스 철교 + 건너편 도시 | `side` `along` `aerial` `bank` | `track_1` `track_2` `bank` `deck` |
| `apartments` | 아파트 단지 | `aerial` `low` `rise` | `lawn` `path` |
| `room` | 방 안 (침대·책상·창문) | `inside` `close` | `floor` `bed` |
| `field` | 넓은 잔디밭 + 먼 스카이라인 | `forward` `low` `aerial` | `field` `near` `path` |
| `planet` | 화성/달 표면 + 착륙선 (`params: {surface: moon}`) | `forward` `aerial` `lander` | `surface` `near_lander` `front_of_lander` |

`area`에는 구역 이름 대신 `[x0, z0, x1, z1]` 좌표를 줄 수도 있습니다.

### 군중 옵션
| 옵션 | 뜻 | 예 |
|---|---|---|
| `agent` | `person` `cat` `dog` `car` `train` `astronaut` `tree` `bed` | `cat` |
| `count` | 개수 (수천 개도 빠름) | `3000` |
| `layout` | `random` / `grid`(줄 맞춤) / `queue`(한 줄, `axis: x`로 가로) | `queue` |
| `facing` | `random` / 각도 / `[x, z]`(그 지점을 봄) / `velocity` | `180` |
| `velocity` | 이동 속도 m/s `[vx, vz]`, 구역 안에서 순환 | `[0, 1.8]` |
| `bob`, `bob_freq` | 들썩임·점프 높이(m)와 빠르기 | `1.2`, `2.0` |
| `scale` | 크기 배율 (멀리서도 보이게) | `[1.6, 2.0]` |
| `colors`, `colors2` | 색 팔레트 직접 지정 | `["#e8913a"]` |
| `y` | 높이 고정 (철교 위 열차 등) | `15` |

카메라가 지나가는 길 위의 군중은 자동으로 비켜서 렌즈를 가리지 않습니다.

### 조명 시각
- `hud`가 시계면(`"05:00"` 형식) 하늘·햇빛·안개·창문 불빛이 자동으로 그 시각을 따라갑니다. 새벽은 분홍, 낮은 파랑, 노을, 밤에는 별과 창문 불빛이 켜져요.
- 직접 정하려면 `time: "21:30"` 또는 `time: {from: "18:00", to: "19:30"}`을 씁니다.
- `world: {template: planet, params: {surface: moon}}`은 검은 우주 하늘에 지구가 떠 있고, `surface: mars`는 황토색 하늘입니다.

### 카메라 직접 지정
```yaml
shot: {from: [0, 7, -30], to: [0, 7, 40], target: [0, 5, 220], fov: 55}     # 이동 촬영
shot: {type: orbit, center: [0, 1, 41], radius: 17, height: 3, from_deg: 150, to_deg: 195}
shot: {type: topdown, center: [0, 0, 0], height: 90, height_to: 75}
```
좌표는 미터 단위이고 x는 오른쪽, y는 위, z는 앞쪽입니다.

---

## 4. 그 밖의 기능

| 기능 | 명령 / 설정 |
|---|---|
| 2D 도트 스타일 (이전 방식) | `style: cinematic_pixel` 또는 `retro_dot` — `examples/earth_rotation_stop.yaml` |
| AI 이미지·영상 도트 후처리 | `python -m if60 pixelate 입력.mp4 -o out/결과.mp4` |
| 주제 → 영상 한 번에 | `python -m if60 make "주제"` (위 '가장 쉬운 사용법') |
| 주제로 장면 파일만 만들기 | `python -m if60 plan "주제" -o scenes/초안.yaml` |
| 칩튠 배경음악 | `music: {mood: wonder, bpm: 104, seed: 16}` (wonder/tense/calm/epic) |
| 효과 | `shake`(카메라 흔들림+효과음) `flash` `tint` / `camera: {fade_in: 0.5, fade_out: 0.8}` |

### API 키 (선택 기능에만 필요)
키는 **절대 YAML이나 코드에 적지 말고** 환경변수로만 넣으세요. `.env` 파일은 `.gitignore`로 저장소에서 제외됩니다.
```bash
python -m if60 key                # Anthropic 키 저장/변경 (make·plan용, .env에 저장)
export OPENAI_API_KEY=...         # AI 이미지 레이어 (2D 스타일)
```
대본 작성 모델은 기본 `claude-sonnet-5-5`이고, `.env`에 `IF60_PLANNER_MODEL=claude-opus-5-5`를 넣으면 더 높은 품질(비용 약 2배)로 바꿀 수 있어요.

---

## 5. 구조

```
if60/
  voxel/
    renderer.py   3D 렌더러 (그림자, MSAA, 하늘, 군중 인스턴싱, 카메라 경로 비우기)
    templates.py  장소 템플릿 8종 + 샷 + 군중 구역
    models.py     블록 모델 (사람, 고양이, 강아지, 자동차, 지하철, 우주비행사, 나무, 침대)
    shaders.py    픽셀 질감 재질, 밤에 켜지는 창문, 물결, 픽셀 해·별
    sky.py        시간대별 하늘·조명
    camera.py     카메라 경로 (이동, 궤도, 부감)
    geometry.py   상자 기반 지오메트리
  render.py      장면 → 영상 파이프라인 (자막, HUD, 효과, 오디오)
  text.py        자막·HUD 배지
  chiptune.py    배경음악·효과음
  pixelate.py    2D 도트 후처리
config/styles.yaml   스타일 프리셋 (voxel, cinematic_pixel, retro_dot)
examples/            예제 장면
```

## 6. 운영 시 주의

- **사실 확인:** 예제 YAML 맨 위의 `facts_to_check`처럼, 대본의 수치는 올리기 전에 반드시 확인하세요.
- **수익 정책:** YouTube는 템플릿으로 찍어낸 듯 차이 없는 대량 생산 콘텐츠를 수익 창출에서 제외합니다. 도구는 제작 노동을 줄이는 데 쓰고, 편마다 주제·관점·연출은 직접 다듬으세요.

## 로드맵
- [x] 3D 복셀 렌더러, 장소 템플릿 8종, 시간대 조명, 군중 인스턴싱
- [x] 시계·카운터 HUD, 끊어 읽는 자막
- [ ] 한국어 TTS 내레이션 + 음성에 맞춘 자막 타이밍 + 배경음악 자동 볼륨 조절
- [ ] 걷기 애니메이션(다리 움직임), 군중 모이기·흩어지기 동작
- [x] 주제 입력 → 웹 조사 → 대본·장면 자동 생성 → 렌더링 (`make`, 더블클릭 실행)
- [ ] 커버(썸네일) 템플릿 자동 생성

## 라이선스
MIT. 폰트·외부 AI 생성물의 라이선스는 각 제공처 약관을 따릅니다.
