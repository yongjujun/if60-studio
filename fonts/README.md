# 폰트

이 폴더에 한글 도트 폰트 파일(.ttf/.otf)을 넣으면 자막에 자동으로 쓰입니다.

추천: **갈무리체 (Galmuri)** — SIL Open Font License 1.1, 상업적 사용 가능
- https://github.com/quiple/galmuri 에서 `Galmuri11.ttf` 등을 받아 이 폴더에 넣으세요.
- 비트맵 폰트는 정해진 크기에서 가장 또렷합니다. Galmuri11은 `caption_size: 22`(2배)처럼 11의 배수를 쓰세요.

폰트가 없으면 OS 기본 한글 폰트(Noto Sans CJK, Apple SD 고딕 Neo, 맑은 고딕)를 찾아 도트처럼 처리합니다.
폰트 파일은 `.gitignore`로 저장소에서 제외되어 있습니다.
