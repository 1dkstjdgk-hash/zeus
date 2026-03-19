# GitHub Pages Deploy

## 개요
- 이 프로젝트는 `docs/` 폴더를 GitHub Pages 배포 소스로 사용합니다.
- `py -3 zeus_integrated_dashboard.py` 를 실행하면 `docs/index.html` 과 필요한 저장 결과 파일이 자동으로 갱신됩니다.
- GitHub Pages는 정적 호스팅이라 방문자 실행형 Python 기능은 지원하지 않습니다.

## 운영 순서
1. 운영자가 로컬 또는 서버에서 필요한 결과를 먼저 생성합니다.
2. `py -3 zeus_integrated_dashboard.py` 를 실행합니다.
3. `docs/` 폴더 갱신 여부를 확인합니다.
4. GitHub 저장소에 커밋 후 `push` 합니다.
5. GitHub 저장소의 `Settings > Pages` 에서 배포 소스를 `Deploy from a branch` 로 선택합니다.
6. 브랜치는 배포 브랜치, 폴더는 `/docs` 로 설정합니다.

## 포함되는 배포 파일
- `docs/index.html`
- `docs/zeus_integrated_out_dashboard.html`
- `docs/zeus_integrated_out_analyzer.html`
- `docs/zeus_integrated_out_backtest.html`
- `docs/zeus_signals.json`
- `docs/zeus_regime.json`
- `docs/zeus_positions.json`
- `docs/screener_results.json`
- `docs/.nojekyll`

## 주의
- GitHub Pages 배포본에서는 `Analyzer`, `Backtest`, `Screener`, `Trading` 모두 저장된 결과만 표시됩니다.
- 공개 실행형 `Analyzer` 가 꼭 필요하면 GitHub Pages 대신 Python 서버 배포가 가능한 호스팅(Render, Railway, VPS 등)이 필요합니다.
