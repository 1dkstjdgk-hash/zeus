# ZEUS Integrated Dashboard

GitHub Pages에 올릴 수 있도록 정리된 ZEUS 통합 대시보드입니다.

## 로컬 갱신
```powershell
py -3 zeus_integrated_dashboard.py
```

실행 후 아래 폴더가 GitHub Pages 배포본으로 갱신됩니다.
- `docs/index.html`
- `docs/*.html`
- `docs/*.json`

## GitHub Pages 설정
1. 이 폴더를 GitHub 저장소에 올립니다.
2. `Settings > Pages` 로 이동합니다.
3. `Build and deployment` 에서 `Source` 를 `Deploy from a branch` 로 선택합니다.
4. 브랜치는 `main` 또는 원하는 배포 브랜치를 선택합니다.
5. 폴더는 `/docs` 를 선택합니다.
6. 저장 후 배포 완료를 기다립니다.

## 커스텀 도메인
- 도메인이 있다면 `docs/CNAME` 파일에 도메인만 한 줄로 넣으면 됩니다.
- 아직 도메인이 없으면 기본 `github.io` 주소로 먼저 운영하면 됩니다.

## 참고
- GitHub Pages는 정적 사이트만 배포합니다.
- 그래서 이 배포본은 저장된 결과를 보여주는 공개 뷰어 용도입니다.
