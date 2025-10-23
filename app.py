from flask import Flask, request, jsonify, render_template
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

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini API 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
    print("Render Dashboard → Environment에서 GOOGLE_API_KEY를 설정해주세요.")
else:
    print("✅ Google API Key가 설정되었습니다.")
    genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Flash 2.5 모델 설정 (무료 버전)
model = genai.GenerativeModel('gemini-2.5-flash')

# 대화 히스토리 저장
chat_sessions = {}

# RAG 문서 저장소
rag_documents = {}

# 문서 인덱스 (빠른 검색을 위한 키워드 매핑)
document_index = {
    'keywords': {},  # 키워드 -> 문서 리스트
    'categories': {},  # 카테고리 -> 문서 리스트
    'entities': {}  # 개체명 -> 문서 리스트
}

# 응답 캐시 (자주 묻는 질문에 대한 빠른 응답)
response_cache = {}

def load_rag_documents():
    """documents/raw 폴더의 모든 파일을 로드하여 RAG 시스템에 저장"""
    global rag_documents
    rag_documents = {}
    
    # documents/raw 폴더의 모든 파일 찾기
    raw_folder = Path('documents/raw')
    if not raw_folder.exists():
        return
    
    # 지원하는 파일 형식
    supported_extensions = ['.txt', '.md', '.csv', '.json', '.ipynb']
    
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
    """스마트 검색: 인덱스 기반 빠른 검색"""
    if not rag_documents:
        return []
    
    query_lower = query.lower()
    relevant_docs = []
    
    # 1. 키워드 기반 검색 (인덱스 활용)
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
            
            # 신한카드 데이터는 더 많은 문자 사용 (제한 강화)
            max_chars = 2000 if '신한카드' in filename.lower() or 'shinhan' in filename.lower() else 500
            
            relevant_docs.append({
                'filename': filename,
                'content': doc_info['content'][:max_chars],
                'relevance_score': relevance_score
            })
    
    # 관련도 순으로 정렬하고 상위 문서만 반환
    relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_docs[:max_docs]

# 앱 시작 시 RAG 문서 로드 및 인덱스 구축
load_rag_documents()
build_document_index()
print(f"📚 문서 로드 완료: {len(rag_documents)}개")
print(f"🔍 인덱스 구축 완료: {len(document_index['keywords'])}개 키워드")

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
        common_events_df = pd.read_csv('documents/raw/성동구 공통_한양대_흥행영화 이벤트 DB2.csv', encoding='utf-8')
        
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
        popup_events_df = pd.read_csv('documents/raw/성수 팝업 최종2.csv', encoding='utf-8')
        
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
    return render_template('index.html')

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
        
        # 세션별 채팅 히스토리 관리
        if session_id not in chat_sessions:
            chat_sessions[session_id] = model.start_chat(history=[])
        
        chat = chat_sessions[session_id]
        
        # 메뉴별 프롬프트 정의
        def get_system_prompt(menu_type=None):
            base_context = """당신은 성동구 지역 소상공인을 위한 전문 마케팅 도우미입니다.

성동구 지역 정보:
- 성수동: 트렌디한 카페, 팝업스토어, 젊은 층 중심
- 왕십리: 전통시장, 중앙시장, 전통과 현대 공존
- 응봉동: 주거지역, 가족 중심
- 옥수동: 한강 근처, 레저업종 유리
- 금호동: 전통 상업지역

답변 형식:
- 마크다운 형식으로 구조화된 답변
- 제목, 목록, 강조 등을 활용
- 구체적이고 실행 가능한 조언"""

            if menu_type == "지역마케팅":
                return base_context + """

당신의 역할 (지역 마케팅 전문):
1. 성동구 각 동네별 특성을 분석한 맞춤형 마케팅 전략
2. 지역 상권 분석 및 경쟁업체 대응 방안
3. 지역 커뮤니티와의 연계 방안
4. 지역 특화 이벤트 및 프로모션 아이디어
5. 지역 주민 대상 타겟팅 전략"""

            elif menu_type == "SNS마케팅":
                return base_context + """

당신의 역할 (SNS 마케팅 전문):
1. 인스타그램, 블로그, 페이스북 등 플랫폼별 전략
2. 해시태그 및 콘텐츠 기획 조언
3. 인플루언서 협업 및 UGC 전략
4. SNS 광고 및 부스팅 전략
5. 바이럴 마케팅 및 트렌드 활용법"""

            elif menu_type == "저예산홍보":
                return base_context + """

당신의 역할 (저예산 홍보 전문):
1. 무료/저비용 마케팅 채널 활용법
2. 오프라인 홍보 전략 (전단지, 현수막, 입간판 등)
3. 지역 이벤트 및 협업 기회 활용
4. 입소문 마케팅 전략
5. 고객 추천 프로그램 및 리워드 시스템"""

            elif menu_type == "이벤트기획":
                return base_context + """

당신의 역할 (이벤트 기획 전문):
1. 고객 유치를 위한 창의적 이벤트 아이디어
2. 계절별/테마별 이벤트 기획
3. 이벤트 홍보 및 참여 유도 전략
4. 이벤트 성과 측정 및 개선 방안
5. 협업 이벤트 및 지역 연계 방안"""

            else:
                return base_context + """

당신의 역할:
1. 성동구 지역 특성을 고려한 맞춤형 마케팅 조언
2. 예산별 실용적인 전략 제안
3. SNS, 오프라인, 이벤트 등 다양한 마케팅 방법 안내
4. 마케팅 템플릿과 구체적인 실행 방법 제공"""

        # 메뉴 타입 확인
        menu_type = data.get('menu_type', None)
        system_context = get_system_prompt(menu_type)
        
        # RAG: 관련 문서 검색
        relevant_docs = search_relevant_documents(user_message)
        
        # RAG 컨텍스트 추가
        rag_context = ""
        if relevant_docs:
            rag_context = "\n\n=== 📊 신한카드 데이터 분석 결과 및 참고 문서 ===\n"
            for doc in relevant_docs:
                rag_context += f"\n📁 파일: {doc['filename']}\n"
                rag_context += f"📋 내용: {doc['content']}\n"
                rag_context += "---\n"
            rag_context += "\n🔍 **중요**: 위 신한카드 데이터 분석 결과를 반드시 참고하여 구체적이고 데이터 기반의 마케팅 전략을 제안해주세요.\n"
            rag_context += "💡 특히 상권별 특성, 고객층 분석, 시간대별 패턴, 업종별 인사이트를 활용하여 실무진이 바로 실행할 수 있는 전략을 제시해주세요.\n"
            rag_context += "📊 **답변 형식**: 답변할 때는 반드시 '신한카드 데이터 정보를 바탕으로 말씀드리자면...', '분석 결과에 따르면...', '데이터에서 확인된 바에 따르면...' 등의 표현을 사용하여 데이터 출처를 명시해주세요.\n"
        
        # 첫 메시지에 시스템 컨텍스트와 RAG 컨텍스트 추가
        if len(chat.history) == 0:
            full_message = f"{system_context}{rag_context}\n\n사용자: {user_message}"
        else:
            full_message = f"{rag_context}\n\n사용자: {user_message}"
        
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
            # API Key 확인
            if not GOOGLE_API_KEY:
                return jsonify({
                    'message': '❌ Google API Key가 설정되지 않았습니다. 관리자에게 문의하세요.',
                    'session_id': session_id
                }), 500
            
            direct_model = genai.GenerativeModel('gemini-2.5-flash')
            
            # RAG 컨텍스트를 포함한 완전한 프롬프트 (길이 제한)
            full_prompt = f"{system_context}{rag_context}\n\n사용자 질문: {user_message}"
            
            # 프롬프트 길이 제한 (너무 길면 잘라내기)
            if len(full_prompt) > 8000:
                full_prompt = f"{system_context}\n\n사용자 질문: {user_message}"
                print("⚠️ 프롬프트가 너무 길어서 RAG 컨텍스트를 제외했습니다.")
            
            # 타임아웃 설정 (30초) - threading 방식
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def api_call():
                try:
                    response = direct_model.generate_content(full_prompt)
                    result_queue.put(('success', response.text))
                except Exception as e:
                    result_queue.put(('error', str(e)))
            
            # API 호출을 별도 스레드에서 실행
            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()
            
            # 120초 타임아웃으로 결과 대기 (2분)
            try:
                result_type, result_data = result_queue.get(timeout=120)
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
        print(f"Error: {str(e)}")
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

@app.route('/api/calendar-events', methods=['GET'])
def get_calendar_events():
    """달력 이벤트 데이터 API"""
    try:
        events = load_event_data()
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': f'이벤트 데이터 로드 실패: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify({'status': 'ok', 'service': 'seongdong-marketing-helper'}), 200

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
    pass

