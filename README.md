# 성동구 마케팅 도우미

성동구 지역 소상공인을 위한 AI 기반 마케팅 도우미 서비스입니다.

## 🚀 Render 배포 가이드

### 1. GitHub에 코드 푸시
```bash
git add .
git commit -m "feat: render 배포 준비"
git push origin main
```

### 2. Render에서 배포
1. [Render.com](https://render.com) 가입
2. **New** → **Web Service** 클릭
3. GitHub Repository 연결
4. 설정:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Port**: 10000 (자동 설정)

### 3. 환경변수 설정
Render Dashboard → Environment에서 추가:
- `GOOGLE_API_KEY`: Google Gemini API 키
- `FLASK_ENV`: production
- `PORT`: 10000

### 4. 배포 완료
- 자동으로 HTTPS URL 제공
- 도메인 설정 가능
- 자동 재배포 (GitHub 푸시 시)

## 📁 프로젝트 구조
```
├── app.py                 # Flask 애플리케이션
├── requirements.txt       # Python 의존성
├── Procfile             # Render 배포 설정
├── render.yaml          # Render 설정 파일
├── templates/
│   └── index.html       # 메인 페이지
├── documents/raw/       # RAG 문서
└── static/             # 정적 파일
```

## 🛠️ 로컬 개발
```bash
# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export GOOGLE_API_KEY=your_api_key

# 서버 실행
python app.py
```

## 📚 주요 기능
- AI 기반 마케팅 조언
- 성동구 지역 특화 전략
- RAG 시스템을 통한 데이터 기반 답변
- 달력 이벤트 관리
- 메뉴별 맞춤 프롬프트
