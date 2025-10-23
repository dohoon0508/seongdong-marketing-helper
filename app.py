from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import pandas as pd
from datetime import datetime, timedelta
import glob
import re
from pathlib import Path
import csv
import json

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini API 설정 (지연 로딩)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
_model = None  # 지연 로딩을 위한 전역 변수

def get_model():
    """지연 로딩으로 모델 가져오기"""
    global _model
    if _model is None:
        if not GOOGLE_API_KEY:
            print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("Render Dashboard → Environment에서 GOOGLE_API_KEY를 설정해주세요.")
            return None
        else:
            print("✅ Google API Key가 설정되었습니다.")
            genai.configure(api_key=GOOGLE_API_KEY)
            # Gemini Flash 2.5 모델 설정 (무료 버전)
            _model = genai.GenerativeModel('gemini-2.5-flash')
            print("🤖 Gemini 모델 로드 완료")
    return _model

# 대화 히스토리 저장
chat_sessions = {}

# RAG 문서 저장소 (지연 로딩)
_rag_documents = None
_document_index = None
_response_cache = {}

# 전역 변수들
rag_documents = {}
document_index = {}
response_cache = {}

def get_rag_documents():
    """지연 로딩으로 RAG 문서 가져오기"""
    global _rag_documents, rag_documents
    if _rag_documents is None:
        print("📚 RAG 문서 로딩 시작...")
        _rag_documents = load_rag_documents()
        rag_documents = _rag_documents  # 전역 변수도 업데이트
        print(f"📚 문서 로드 완료: {len(_rag_documents)}개")
    return _rag_documents

def get_document_index():
    """지연 로딩으로 문서 인덱스 가져오기"""
    global _document_index
    if _document_index is None:
        print("🔍 문서 인덱스 구축 시작...")
        _document_index = build_document_index()
        print(f"🔍 인덱스 구축 완료: {len(_document_index['keywords'])}개 키워드")
    return _document_index

def load_rag_documents():
    """documents/raw 폴더의 모든 파일을 로드하여 RAG 시스템에 저장"""
    global rag_documents
    rag_documents = {}
    
    # documents/raw 폴더의 모든 파일 찾기
    raw_folder = Path('documents/raw')
    if not raw_folder.exists():
        return rag_documents
    
    # 지원하는 파일 형식
    supported_extensions = ['.txt', '.md', '.csv', '.json', '.jsonl', '.ipynb']
    
    for file_path in raw_folder.glob('*'):
        if file_path.suffix.lower() in supported_extensions:
            try:
                if file_path.suffix.lower() == '.csv':
                    # CSV 파일 처리
                    df = pd.read_csv(file_path, encoding='utf-8')
                    content = f"파일명: {file_path.name}\n\n"
                    content += f"데이터 형태: CSV\n"
                    content += f"행 수: {len(df)}\n"
                    content += f"열: {', '.join(df.columns.tolist())}\n\n"
                    
                    # 처음 5행의 데이터 샘플
                    content += "데이터 샘플:\n"
                    content += df.head().to_string()
                    
                elif file_path.suffix.lower() == '.md':
                    # Markdown 파일 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                elif file_path.suffix.lower() == '.txt':
                    # 텍스트 파일 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                elif file_path.suffix.lower() == '.json':
                    # JSON 파일 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                elif file_path.suffix.lower() == '.jsonl':
                    # JSONL 파일 처리
                    import json
                    content = f"파일명: {file_path.name}\n\n"
                    content += f"데이터 형태: JSONL (JSON Lines)\n\n"
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    content += f"총 라인 수: {len(lines)}\n\n"
                    content += "데이터 내용:\n"
                    
                    # 신한카드분석.jsonl은 전체 레코드 처리 (중요한 데이터)
                    if '신한카드분석.jsonl' in file_path.name:
                        max_records = len(lines)  # 전체 레코드 처리
                    else:
                        max_records = 5  # 다른 파일은 메모리 최적화
                    for i, line in enumerate(lines[:max_records]):
                        if line.strip():  # 빈 줄이 아닌 경우만
                            try:
                                data = json.loads(line.strip())
                                content += f"--- 레코드 {i+1} ---\n"
                                for key, value in data.items():
                                    content += f"{key}: {value}\n"
                                content += "\n"
                            except json.JSONDecodeError:
                                content += f"--- 레코드 {i+1} (JSON 파싱 오류) ---\n{line.strip()}\n\n"
                    
                    if len(lines) > max_records:
                        content += f"... (총 {len(lines)}개 레코드 중 {max_records}개만 표시)\n"
                        
                elif file_path.suffix.lower() == '.ipynb':
                    # Jupyter Notebook 파일 처리
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook = json.load(f)
                    
                    content = f"파일명: {file_path.name}\n\n"
                    content += f"노트북 타입: Jupyter Notebook\n"
                    content += f"셀 수: {len(notebook.get('cells', []))}\n\n"
                    
                    # 각 셀의 내용 추출
                    for i, cell in enumerate(notebook.get('cells', [])):
                        cell_type = cell.get('cell_type', 'unknown')
                        source = ''.join(cell.get('source', []))
                        
                        if cell_type == 'markdown':
                            content += f"## 셀 {i+1} (마크다운):\n{source}\n\n"
                        elif cell_type == 'code':
                            content += f"## 셀 {i+1} (코드):\n```python\n{source}\n```\n\n"
                        elif cell_type == 'raw':
                            content += f"## 셀 {i+1} (텍스트):\n{source}\n\n"
                
                # 문서 저장
                rag_documents[file_path.name] = {
                    'content': content,
                    'file_type': file_path.suffix.lower(),
                    'file_path': str(file_path)
                }
                
            except Exception as e:
                print(f"파일 로드 오류 {file_path.name}: {e}")
                continue
    
    return rag_documents

def build_document_index():
    """문서 인덱스 구축 (키워드, 카테고리, 개체명 추출)"""
    global document_index
    document_index = {'keywords': {}, 'categories': {}, 'entities': {}}
    
    # 키워드 추출 및 매핑
    for filename, doc_info in rag_documents.items():
        content = doc_info['content'].lower()
        
        # 키워드 추출 (간단한 형태소 분석)
        keywords = extract_keywords(content)
        for keyword in keywords:
            if keyword not in document_index['keywords']:
                document_index['keywords'][keyword] = []
            document_index['keywords'][keyword].append(filename)
        
        # 카테고리 분류
        category = classify_document(filename, content)
        if category not in document_index['categories']:
            document_index['categories'][category] = []
        document_index['categories'][category].append(filename)
    
    return document_index

def extract_keywords(text):
    """텍스트에서 중요한 키워드 추출"""
    # 간단한 키워드 추출 (실제로는 더 정교한 NLP 필요)
    important_words = ['팝업', '이벤트', '성동구', '성수', '마케팅', '고객', '매출', '분석', '데이터']
    found_keywords = []
    for word in important_words:
        if word in text:
            found_keywords.append(word)
    return found_keywords

def classify_document(filename, content):
    """문서 카테고리 분류"""
    if '신한카드' in filename.lower():
        return '데이터분석'
    elif '팝업' in filename.lower() or '이벤트' in filename.lower():
        return '이벤트'
    elif '마케팅' in content:
        return '마케팅'
    else:
        return '기타'

def search_relevant_documents(query, max_docs=3):
    """스마트 검색: 인덱스 기반 빠른 검색 (지연 로딩)"""
    # 지연 로딩으로 문서와 인덱스 가져오기
    rag_documents = get_rag_documents()
    document_index = get_document_index()
    
    if not rag_documents or rag_documents is None:
        return []
    
    query_lower = query.lower()
    relevant_docs = []
    
    # 1. 신한카드분석.jsonl 파일 우선 처리
    shinhan_file = None
    for filename in rag_documents.keys():
        if '신한카드분석.jsonl' in filename:
            shinhan_file = filename
            break
    
    # 모든 질문에 대해 신한카드분석.jsonl을 최우선으로 포함
    if shinhan_file:
        doc_info = rag_documents[shinhan_file]
        relevant_docs.append({
            'filename': shinhan_file,
            'content': doc_info['content'][:8000],  # 신한카드 파일은 더 많은 문자 사용 (3000 → 8000)
            'relevance_score': 1000  # 최고 우선순위 (기존 100에서 1000으로 증가)
        })
    
    # 2. 키워드 기반 검색 (인덱스 활용)
    candidate_files = set()
    for keyword in query_lower.split():
        if keyword in document_index['keywords']:
            candidate_files.update(document_index['keywords'][keyword])
    
    # 2. 카테고리 기반 검색
    if '팝업' in query_lower or '이벤트' in query_lower:
        if '이벤트' in document_index['categories']:
            candidate_files.update(document_index['categories']['이벤트'])
    
    if '데이터' in query_lower or '분석' in query_lower:
        if '데이터분석' in document_index['categories']:
            candidate_files.update(document_index['categories']['데이터분석'])
    
    # 3. 후보 파일들 중에서 관련도 계산
    for filename in candidate_files:
        if filename in rag_documents:
            doc_info = rag_documents[filename]
            content = doc_info['content'].lower()
            
            # 관련도 점수 계산
            relevance_score = 0
            for keyword in query_lower.split():
                if keyword in content:
                    relevance_score += content.count(keyword)
                if keyword in filename.lower():
                    relevance_score += 2
            
            # 신한카드분석.jsonl 파일에 추가 가중치 (최우선)
            if '신한카드분석.jsonl' in filename:
                relevance_score += 500  # 기존 50에서 500으로 대폭 증가
            
            # 신한카드 데이터는 더 많은 문자 사용 (제한 강화)
            max_chars = 8000 if '신한카드' in filename.lower() or 'shinhan' in filename.lower() else 500
            
            relevant_docs.append({
                'filename': filename,
                'content': doc_info['content'][:max_chars],
                'relevance_score': relevance_score
            })
    
    # 관련도 순으로 정렬하고 상위 문서만 반환
    relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_docs[:max_docs]

# Render 환경에서 앱 시작 시 문서 로딩 비활성화 (메모리 절약)
print("🚀 Render 환경에서 실행 중 - 문서 로딩은 요청 시 처리됩니다.")
rag_documents = {}
document_index = {'keywords': {}, 'categories': {}, 'entities': {}}

@app.route('/api/reload-documents', methods=['POST'])
def reload_documents():
    """문서를 다시 로드하는 API"""
    try:
        load_rag_documents()
        return jsonify({
            'success': True, 
            'message': f'{len(rag_documents)}개의 문서가 로드되었습니다.',
            'documents': list(rag_documents.keys())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """현재 로드된 문서 목록 조회"""
    return jsonify({
        'documents': [
            {
                'filename': filename,
                'file_type': doc_info['file_type'],
                'content_length': len(doc_info['content'])
            }
            for filename, doc_info in rag_documents.items()
        ]
    })

# 이벤트 데이터 로드 함수
def load_event_data():
    events = {}
    
    try:
        # 성동구 공통 이벤트 데이터 로드
        common_events_df = pd.read_csv('documents/raw/성동구 공통_한양대_흥행영화 이벤트 DB.csv', encoding='utf-8')
        
        for _, row in common_events_df.iterrows():
            start_date = pd.to_datetime(row['Start_Date'], format='%Y.%m.%d', errors='coerce')
            end_date = pd.to_datetime(row['End_Date'], format='%Y.%m.%d', errors='coerce')
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            # 이벤트 타입 분류
            event_type = 'general'
            if '팝업' in str(row['Event_Type']):
                event_type = 'popup'
            elif '대학' in str(row['Event_Type']) or '한양대' in str(row['Event_Name']):
                event_type = 'university'
            elif '영화' in str(row['Event_Type']) or '개봉' in str(row['Event_Name']):
                event_type = 'movie'
            
            # 날짜 범위에 따라 이벤트 추가
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in events:
                    events[date_str] = []
                
                events[date_str].append({
                    'name': row['Event_Name'],
                    'type': event_type,
                    'startDate': start_date.strftime('%Y-%m-%d'),
                    'endDate': end_date.strftime('%Y-%m-%d'),
                    'location': row['Location_Address'],
                    'target': row['Target_Audience'],
                    'description': row['Event_Description']
                })
                
                current_date += timedelta(days=1)
                
    except Exception as e:
        print(f"공통 이벤트 데이터 로드 실패: {e}")
    
    try:
        # 성수 팝업 데이터 로드
        popup_events_df = pd.read_csv('documents/raw/성수 팝업 최종.csv', encoding='utf-8')
        
        for _, row in popup_events_df.iterrows():
            start_date = pd.to_datetime(row['Start_Date'], format='%Y.%m.%d', errors='coerce')
            end_date = pd.to_datetime(row['End_Date'], format='%Y.%m.%d', errors='coerce')
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            
            # 날짜 범위에 따라 이벤트 추가
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in events:
                    events[date_str] = []
                
                events[date_str].append({
                    'name': row['Event_Name'],
                    'type': 'popup',
                    'startDate': start_date.strftime('%Y-%m-%d'),
                    'endDate': end_date.strftime('%Y-%m-%d'),
                    'location': row['Location_Address'],
                    'target': row['Target_Audience'],
                    'description': row['Event_Description']
                })
                
                current_date += timedelta(days=1)
                
    except Exception as e:
        print(f"팝업 이벤트 데이터 로드 실패: {e}")
    
    return events

@app.route('/')
def index():
    # 메인 페이지는 설정 페이지로 리다이렉트
    return redirect('/setup')

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        # 저장된 설정 정보 가져오기
        setup_info = user_setups.get('default', None)
        if not setup_info:
            return jsonify({'error': '설정이 완료되지 않았습니다. 먼저 설정을 완료해주세요.'}), 400
        
        location = setup_info['location']
        industry = setup_info['industry']
        store_name = setup_info['store_name']
        
        print(f"📍 지역: {location}, 🏢 업종: {industry}, 🏪 가게: {store_name}")
        
        # 세션별 채팅 히스토리 관리
        if session_id not in chat_sessions:
            model = get_model()
            if model is None:
                return jsonify({
                    'message': '❌ Google API Key가 설정되지 않았습니다. 관리자에게 문의하세요.',
                    'status': 'error'
                }), 500
            chat_sessions[session_id] = model.start_chat(history=[])
        
        chat = chat_sessions[session_id]
        
        # 시스템 프롬프트 생성 (새로운 템플릿)
        system_prompt = f"""[SYSTEM / ROLE]
너는 성동구 소상공인을 위한 맞춤 마케팅 어시스턴트다. 답변은 한국어로, 실행 가능한 TODO를 최우선으로 제시한다.

[CONTEXT PRIORITY]
- 1순위: 신한카드분석.jsonl (키: INS, RULE, POPUP, 상권특성 등)  
- 2순위: 지역/월/업종과 직접 연관된 CSV/JSON (예: "성수 팝업 최종.csv", "성동구 공통_한양대_흥행영화 이벤트 DB.csv", 기타 상권/행사 DB)
- 같은 정보가 중복일 땐 1순위를 우선 채택한다.

[STRICT CITATION]
- 모든 주장/숫자/사실 뒤에 (출처: 파일명[#레코드ID]) 형식으로 근거를 표기한다.
  - 예: (출처: 신한카드분석.jsonl#INS-12, RULE-3), (출처: 성수 팝업 최종.csv#row128)
- 출처가 불명확하면 "데이터 미확인"이라고 명시하고 추정 발화를 하지 않는다.

[RETRIEVAL SCOPE]
- REGION={location}, INDUSTRY={industry}, STORE={store_name}, MONTH=현재월
- 먼저 REGION×INDUSTRY 키로 파티션을 좁혀 L0 프로필 문서를 로드한다.
- 부족할 때만 동일 파티션에서 Top-K=3, 윈도우=400~600자 스니펫으로 L1 슬림 RAG 보강한다.
- 검색어는 `{location} {industry} 현재월 팝업/이벤트/소비패턴/시간대/타깃` 중심으로 확장한다.
- 최신성이 필요한 항목(이달 행사 등)은 최신 월 우선 정렬한다.

[STYLE & TONE]
- 첫 문장 고정: "{location} 지역의 {industry} 사장님, 안녕하세요. 현재월 마케팅 가이드를 정리했습니다."
- 문단은 짧게, 리스트/표를 적극 활용. 사장님이 바로 실행할 수 있도록 수치·행동·툴을 구체화.
- 지역 특징과 업종 특성을 "교집합 관점"으로 제시.

[OUTPUT FORMAT]
# ☕ {location} {industry}를 위한 현재월 맞춤형 마케팅 전략

간단요약(2~3문장) — 핵심 인사이트와 이번 달 기회 포인트. (출처: …)

1. 상권·수요 핵심 포인트
- ● 유동/연령/시간대/객단가 핵심 관찰 3~5개 (숫자/근거 포함). (출처: …)

2. 이번 달( 현재월 ) 이벤트/팝업 연계 아이디어
- ● 아이디어명 — 왜/어떻게/예상효과/간단 실행 절차. (출처: …)
- ● …

3. 채널별 실전 액션(이번 주 바로 실행)
- [ ] 네이버플레이스: 키워드/해시태그/리뷰 리프레이밍(예시 문구). (출처: …)
- [ ] 인스타 릴스: 캘린더 연동/콘텐츠 테마/업로드 시각. (출처: …)
- [ ] 오프라인: 세트/타임세일/콜라보 구체안. (출처: …)

4. 가격·구성 제안(선택)
- ● 점심 회전/저녁 체류형 각 1안씩: 구성/가격/전환 트리거. (출처: …)

5. 근거/출처 목록
- 신한카드분석.jsonl#INS-…, RULE-…, POPUP-…
- 성수 팝업 최종.csv#row…, 성동구 공통_한양대_흥행영화 이벤트 DB.csv#row…"""

        # L0 프로필 로딩
        profile_text = load_l0_profile(location, industry)
        
        # L1 슬림 RAG 검색 (필요시에만)
        snippets = ""
        if needs_more_context(user_message, profile_text):
            snippets = slim_search(user_message, f"{location}:{industry}", 3)
        
        # 프롬프트 조립
        prompt_parts = [system_prompt]
        
        if profile_text:
            prompt_parts.append(f"[프로필]\n{profile_text}")
        
        if snippets:
            prompt_parts.append(f"[보강]\n{snippets}")
        
        prompt_parts.append(f"[사용자 질문]\n{user_message}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # 캐시 확인 (자주 묻는 질문에 대한 빠른 응답)
        cache_key = user_message.lower().strip()
        if cache_key in response_cache:
            print(f"🚀 캐시에서 응답 반환: {cache_key}")
            return jsonify({
                'message': response_cache[cache_key],
                'session_id': session_id
            })
        
        # RAG 기반 응답 (모든 질문에 대해)
        # 성능 모니터링 시작
        import time
        start_time = time.time()
        print(f"🔍 검색 시작: {user_message[:30]}...")
        
        # Gemini API 호출 (RAG 컨텍스트 포함) - 타임아웃 설정
        try:
            # 지연 로딩으로 모델 가져오기
            direct_model = get_model()
            if direct_model is None:
                print("❌ API Key 또는 모델이 설정되지 않음")
                return jsonify({
                    'message': '❌ Google API Key가 설정되지 않았습니다. 관리자에게 문의하세요.',
                    'session_id': session_id
                }), 500
            
            # 프롬프트 길이 제한 (1200~1800 토큰 기준, 약 4000~6000자)
            if len(full_prompt) > 6000:
                # 시스템 프롬프트만 사용하여 길이 제한
                full_prompt = f"{system_prompt}\n\n[사용자 질문]\n{user_message}"
                print("⚠️ 프롬프트가 너무 길어서 RAG 컨텍스트를 제외했습니다.")
            
            # 타임아웃 설정 (30초) - threading 방식
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def api_call():
                try:
                    # 외부 API 호출에 타임아웃 및 재시도 추가
                    import time
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            response = direct_model.generate_content(full_prompt)
                            result_queue.put(('success', response.text))
                            return
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"⚠️ API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                                time.sleep(retry_delay * (attempt + 1))  # 지수 백오프
                            else:
                                result_queue.put(('error', str(e)))
                except Exception as e:
                    result_queue.put(('error', str(e)))
            
            # API 호출을 별도 스레드에서 실행
            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()
            
            # 180초 타임아웃으로 결과 대기 (3분)
            try:
                result_type, result_data = result_queue.get(timeout=180)
                if result_type == 'success':
                    response_text = result_data
                    print(f"✅ Gemini API 응답 성공: {response_text[:50]}...")
                    print(f"📊 프롬프트 길이: {len(full_prompt)}자")
                    print(f"⏱️ 응답 시간: {time.time() - start_time:.2f}초")
                else:
                    raise Exception(result_data)
            except queue.Empty:
                print("⏰ API 타임아웃 (120초 초과)")
                response_text = """## ⏰ 응답 시간 초과

죄송합니다. 요청이 너무 복잡해서 처리 시간이 오래 걸리고 있습니다.

### 🔧 해결 방법:
1. **질문을 더 구체적으로** 말씀해주세요
2. **키워드 중심으로** 간단히 질문해주세요
3. **잠시 후 다시 시도**해주세요

### 💡 빠른 질문 예시:
- "성동구 팝업 알려줘"
- "마케팅 전략 추천해줘"
- "고객 유치 방법 알려줘"

더 간단한 질문으로 다시 시도해주세요! 🚀"""
                
        except TimeoutError:
            print("⏰ API 타임아웃 (180초 초과)")
            response_text = """## ⏰ 응답 시간 초과

죄송합니다. 요청 처리가 예상보다 오래 걸리고 있습니다.

### 🔧 해결 방법:
1. **잠시 후 다시 시도**해주세요 (서버가 바쁠 수 있습니다)
2. **질문을 좀 더 구체적으로** 말씀해주세요
3. **키워드 중심으로** 간단히 질문해주세요

### 💡 빠른 질문 예시:
- "성동구 팝업 알려줘"
- "마케팅 전략 추천해줘"
- "고객 유치 방법 알려줘"

잠시 후 다시 시도해주세요! 🚀"""
        except Exception as api_error:
            print(f"❌ API Error: {str(api_error)}")
            
            # Google API 관련 오류 처리
            if "API_KEY" in str(api_error) or "authentication" in str(api_error).lower():
                response_text = """## 🔑 API 인증 오류

Google API Key가 올바르지 않거나 설정되지 않았습니다.

### 🔧 해결 방법:
1. **Render Dashboard** → **Environment**에서 `GOOGLE_API_KEY` 확인
2. **Google AI Studio**에서 새로운 API Key 생성
3. **관리자에게 문의**하세요

### 📝 API Key 설정 방법:
1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. **Create API Key** 클릭
3. 생성된 키를 Render Environment에 추가
"""
            elif "quota" in str(api_error).lower() or "limit" in str(api_error).lower():
                response_text = """## 📊 API 할당량 초과

Google API 사용량이 한도를 초과했습니다.

### 🔧 해결 방법:
1. **잠시 후 다시 시도**해주세요 (보통 1시간 후 복구)
2. **더 간단한 질문**으로 시도해주세요
3. **관리자에게 문의**하세요

### 💡 빠른 질문 예시:
- "성동구 팝업 알려줘"
- "마케팅 전략 추천해줘"
"""
            else:
                response_text = f"""## ❌ 일시적인 오류 발생

죄송합니다. 시스템에 일시적인 오류가 발생했습니다.

### 🔧 해결 방법:
1. **잠시 후 다시 시도**해주세요
2. **더 간단한 질문**으로 시도해주세요
3. **문제가 지속되면 관리자에게 문의**하세요

### 📝 오류 정보:
```
{str(api_error)}
```

더 간단한 질문으로 다시 시도해주세요! 🚀"""
        
        # response_text가 이미 설정되어 있음 (테스트 응답 또는 API 응답)
        
        # 응답 캐시에 저장 (자주 묻는 질문용)
        cache_key = user_message.lower().strip()
        if len(cache_key) < 100:  # 너무 긴 질문은 캐시하지 않음
            response_cache[cache_key] = response_text
            print(f"💾 캐시에 저장: {cache_key[:20]}...")
        
        # 마크다운을 HTML로 변환
        html_response = markdown.markdown(response_text, extensions=['extra'])
        
        return jsonify({
            'message': html_response,
            'session_id': session_id
        })
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/test-gemini', methods=['GET'])
def test_gemini():
    """Gemini API 직접 테스트"""
    try:
        direct_model = genai.GenerativeModel('gemini-2.5-flash')
        response = direct_model.generate_content('안녕하세요, 성동구 소상공인 마케팅 도우미입니다.')
        response_text = response.text
        html_response = markdown.markdown(response_text, extensions=['extra'])
        return jsonify({'response': html_response, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        return jsonify({'message': '대화가 초기화되었습니다.'})
    
    except Exception as e:
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify({'status': 'ok'}), 200

# ====== 설정 관리 시스템 ======

# 전역 설정 저장소 (실제로는 데이터베이스나 세션에 저장해야 함)
user_setups = {}
preloaded_documents = []

@app.route('/api/check-setup', methods=['GET'])
def check_setup():
    """설정이 완료되었는지 확인"""
    try:
        # 세션 기반으로 설정 확인 (실제로는 더 안전한 방법 사용)
        setup_info = user_setups.get('default', None)
        
        if setup_info:
            return jsonify({
                'is_setup': True,
                'setup_info': setup_info
            })
        else:
            return jsonify({
                'is_setup': False
            })
    except Exception as e:
        return jsonify({
            'is_setup': False,
            'error': str(e)
        }), 500

@app.route('/api/setup', methods=['POST'])
def save_setup():
    """설정 정보 저장 및 문서 미리 로딩"""
    try:
        data = request.json
        location = data.get('location', '')
        industry = data.get('industry', '')
        store_name = data.get('store_name', '')
        
        if not location or not industry or not store_name:
            return jsonify({'error': '모든 정보를 입력해주세요.'}), 400
        
        # 설정 정보 저장
        setup_info = {
            'location': location,
            'industry': industry,
            'store_name': store_name,
            'created_at': datetime.now().isoformat()
        }
        
        user_setups['default'] = setup_info
        
        # L0 프로필 미리 로딩
        try:
            profile_text = load_l0_profile(location, industry)
            print(f"📋 L0 프로필 미리 로딩 완료: {location}_{industry}")
        except Exception as e:
            print(f"⚠️ L0 프로필 로딩 실패: {e}")
        
        # 관련 문서 미리 로딩 (신한카드 데이터 등)
        try:
            preload_relevant_documents(location, industry)
            print(f"📚 관련 문서 미리 로딩 완료: {location}_{industry}")
        except Exception as e:
            print(f"⚠️ 문서 미리 로딩 실패: {e}")
        
        return jsonify({
            'success': True,
            'message': '설정이 완료되었습니다.',
            'setup_info': setup_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get-setup', methods=['GET'])
def get_setup():
    """저장된 설정 정보 조회"""
    try:
        setup_info = user_setups.get('default', None)
        
        if setup_info:
            return jsonify({
                'success': True,
                'setup_info': setup_info
            })
        else:
            return jsonify({
                'success': False,
                'error': '설정 정보가 없습니다.'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def preload_relevant_documents(location, industry):
    """관련 문서 미리 로딩"""
    try:
        # 신한카드 데이터에서 관련 스니펫 미리 검색
        shinhan_file = "documents/raw/신한카드분석.jsonl"
        if os.path.exists(shinhan_file):
            relevant_docs = []
            with open(shinhan_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data.get('body', '')
                        
                        # 지역 및 업종 관련성 체크
                        if (location and location in content) or (industry and industry in content):
                            relevant_docs.append({
                                'content': content,
                                'filename': '신한카드분석.jsonl',
                                'location': location,
                                'industry': industry
                            })
                    except:
                        continue
            
            # 전역 변수에 저장 (실제로는 캐시나 데이터베이스에 저장)
            global preloaded_documents
            preloaded_documents = relevant_docs
            print(f"📚 미리 로딩된 문서: {len(relevant_docs)}개")
            
    except Exception as e:
        print(f"⚠️ 문서 미리 로딩 실패: {e}")

# ====== L0 프로필 및 L1 슬림 RAG 시스템 ======

def load_l0_profile(location, industry):
    """L0 프로필 문서 로딩 (지역×업종별 맞춤 프로필)"""
    try:
        # 프로필 파일 경로 생성
        profile_path = f"documents/profiles/{location}_{industry}.md"
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"📋 L0 프로필 로드: {location}_{industry}")
            return content
        
        # 기본 프로필 생성 (파일이 없는 경우)
        default_profile = generate_default_profile(location, industry)
        print(f"📋 기본 L0 프로필 생성: {location}_{industry}")
        return default_profile
        
    except Exception as e:
        print(f"⚠️ L0 프로필 로딩 실패: {e}")
        return generate_default_profile(location, industry)

def generate_default_profile(location, industry):
    """기본 L0 프로필 생성"""
    return f"""# {location} 지역 {industry} 업종 프로필

## 지역 특성
- {location} 지역의 상권 특성 및 고객층 분석
- 주변 경쟁업체 현황
- 접근성 및 교통편

## 업종별 인사이트
- {industry} 업종의 {location} 지역 적합도
- 타겟 고객층 특성
- 성공 사례 및 실패 요인

## 마케팅 전략
- 지역 맞춤형 홍보 방법
- 고객 유치 전략
- 가격 정책 및 서비스 개선 방안

## 참고 데이터
- 신한카드 데이터 기반 분석 결과
- 지역 이벤트 및 프로모션 정보"""

def needs_more_context(user_message, profile_text):
    """L1 슬림 RAG 보강이 필요한지 판단"""
    # 간단한 키워드 기반 판단
    need_keywords = ['구체적', '상세한', '자세한', '세부', '분석', '데이터', '통계', '비교', '경쟁사']
    return any(keyword in user_message for keyword in need_keywords)

def slim_search(user_message, partition, top_k=3):
    """L1 슬림 RAG 검색 (우선순위 기반 검색)"""
    try:
        # 파티션 파싱 (예: "성수:카페")
        if ':' in partition:
            region, industry = partition.split(':', 1)
        else:
            region, industry = "", ""
        
        # 검색어 확장
        search_terms = [region, industry, "팝업", "이벤트", "소비패턴", "시간대", "대학", "영화", "성수기"]
        search_query = " ".join([term for term in search_terms if term])
        
        relevant_snippets = []
        
        # 1순위: 신한카드분석.jsonl에서 INS/RULE/POPUP/상권특성 검색
        shinhan_file = "documents/raw/신한카드분석.jsonl"
        if os.path.exists(shinhan_file):
            with open(shinhan_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        content = data.get('body', '')
                        doc_type = data.get('doc_type', '')
                        title = data.get('title', '')
                        
                        # INS, RULE, POPUP, 상권특성 키워드 우선 검색
                        if any(keyword in doc_type.lower() or keyword in title.lower() 
                               for keyword in ['ins', 'rule', 'popup', '상권특성']):
                            if any(term in content.lower() for term in search_terms if term):
                                snippet = content[:600]  # 윈도우 400~600자
                                source_tag = f"신한카드분석.jsonl#{doc_type}-{line_num}"
                                relevant_snippets.append({
                                    'content': snippet,
                                    'source': source_tag,
                                    'priority': 1,
                                    'score': 1.0
                                })
                        
                        # 일반적인 관련성 체크 (신한카드 데이터)
                        relevance_score = 0
                        if region and region in content:
                            relevance_score += 2
                        if industry and industry in content:
                            relevance_score += 2
                        
                        if relevance_score > 0:
                            snippet = content[:600]
                            source_tag = f"신한카드분석.jsonl#{doc_type}-{line_num}"
                            relevant_snippets.append({
                                'content': snippet,
                                'source': source_tag,
                                'priority': 1,
                                'score': relevance_score
                            })
                    except:
                        continue
        
        # 2순위: CSV 파일들에서 검색 (상위 3개만)
        csv_files = [
            ("성수 팝업 최종.csv", 2),
            ("성동구 공통_한양대_흥행영화 이벤트 DB.csv", 2)
        ]
        
        for csv_file, priority in csv_files:
            csv_path = os.path.join(app.root_path, 'documents', 'raw', csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    for idx, row in df.iterrows():
                        row_content = " ".join([str(v) for v in row.values if pd.notna(v)])
                        if any(term in row_content.lower() for term in search_terms if term):
                            snippet = row_content[:600]
                            source_tag = f"{csv_file}#row{idx+1}"
                            relevant_snippets.append({
                                'content': snippet,
                                'source': source_tag,
                                'priority': priority,
                                'score': 0.8
                            })
                except Exception as e:
                    print(f"⚠️ CSV 파일 처리 실패 {csv_file}: {e}")
        
        # 우선순위와 점수로 정렬 (1순위 신한카드 > 2순위 CSV)
        relevant_snippets.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
        
        # 중복 제거 (유사도 0.9 이상)
        unique_snippets = []
        for snippet in relevant_snippets:
            is_duplicate = False
            for existing in unique_snippets:
                if len(set(snippet['content'].split()) & set(existing['content'].split())) / len(set(snippet['content'].split()) | set(existing['content'].split())) > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_snippets.append(snippet)
        
        # Top-K 선택
        selected_snippets = unique_snippets[:top_k]
        
        # 스니펫 포맷팅
        if selected_snippets:
            formatted_snippets = []
            for i, snippet in enumerate(selected_snippets, 1):
                formatted_snippets.append(f"[스니펫 {i}] (출처: {snippet['source']})\n{snippet['content']}")
            
            result = "\n\n".join(formatted_snippets)
            print(f"🔍 L1 슬림 RAG 검색 완료: {len(selected_snippets)}개 스니펫 (우선순위: {[s['priority'] for s in selected_snippets]})")
            return result
        
        return ""
        
    except Exception as e:
        print(f"⚠️ L1 슬림 RAG 검색 실패: {e}")
        return ""

@app.route('/api/calendar-events', methods=['GET'])
def get_calendar_events():
    """달력 이벤트 데이터 반환"""
    try:
        # CSV 파일 경로
        csv_file_path = os.path.join(app.root_path, 'documents', 'raw', '성수 팝업 최종.csv')
        
        if not os.path.exists(csv_file_path):
            return jsonify({'error': 'CSV 파일이 존재하지 않습니다.'}), 404
        
        events = []
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                events.append(row)
        
        return jsonify({'events': events})
    except Exception as e:
        print(f"Error loading calendar events: {e}")
        return jsonify({'error': '이벤트 로드 중 오류가 발생했습니다.', 'details': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Flask 서버 시작: 0.0.0.0:{port}")
    print(f"📊 환경변수 PORT: {os.environ.get('PORT', 'Not set')}")
    app.run(debug=False, host='0.0.0.0', port=port)
else:
    # Render에서 Gunicorn으로 실행될 때는 이 부분이 실행되지 않음
    # Gunicorn이 직접 app 객체를 import하여 사용
    print("🔧 Gunicorn 모드로 실행 중...")
    
    # Render 환경에서 메모리 절약을 위해 앱 시작 시 문서 로딩 비활성화
    print("🚀 Render 환경에서 실행 중 - RAG 문서는 요청 시 로딩됩니다.")
    rag_documents = {}
    document_index = {'keywords': {}, 'categories': {}, 'entities': {}}
    pass

