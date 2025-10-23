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
        
        # 지역과 업종 정보 가져오기
        location = data.get('location', '')
        industry = data.get('industry', '')
        print(f"📍 지역: {location}, 🏢 업종: {industry}")
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
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
        
        # 메뉴별 프롬프트 정의
        def get_system_prompt(menu_type=None):
            base_context = """당신은 성동구 지역 소상공인을 위한 전문 마케팅 도우미입니다.

❗ **출처 명시 필수 규칙**:
- 모든 답변에는 반드시 데이터 출처를 포함해야 합니다
- 반드시 [출처: 신한카드분석.jsonl] 또는 [출처: 파일명] 형태로 표기하세요

🎯 **핵심 역할**:
- 신한카드 데이터 기반 근거 있는 마케팅 전략 제시
- 지역별/업종별 맞춤형 솔루션 제공
- 재현 가능하고 실행 가능한 구체적 조언

📊 **필수 응답 구조**:
1. **인사말**: 사용자가 선택한 지역과 업종을 반드시 포함하여 "{지역} 지역의 {업종} 사장님을 위한 솔루션을 가져왔습니다."로 시작
2. **근거 기반 분석**: 신한카드 데이터 인용과 함께 상권/업종 특성 분석
3. **구체적 전략**: 실행 가능한 마케팅 전략 3-5가지 제시
4. **출처 명시**: 모든 데이터와 규칙의 출처를 명확히 표기

🚨 **응답 시작 규칙**:
- 사용자가 선택한 지역과 업종 정보가 제공되면, 반드시 해당 정보를 포함한 인사말로 시작해야 합니다
- 일반적인 인사말("성동구 사장님", "안녕하세요" 등)로 시작하면 안됩니다
- 정확한 형식: "{선택된지역} 지역의 {선택된업종} 사장님을 위한 솔루션을 가져왔습니다."

🔍 **데이터 활용 원칙**:
- 신한카드분석.jsonl의 모든 인사이트와 규칙을 적극 활용
- 상권별 특성, 고객층 분석, 시간대별 패턴을 반드시 반영
- 업종별 적합도와 타겟 매칭 규칙을 적용

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
        
        # RAG: 관련 문서 검색 (요청 시 로딩)
        # 문서가 로드되지 않았다면 먼저 로드
        global rag_documents, document_index
        if not rag_documents:
            try:
                print("📚 문서 로딩 시작...")
                rag_documents = load_rag_documents()
                document_index = build_document_index()
                print(f"📚 문서 로드 완료: {len(rag_documents)}개")
            except Exception as e:
                print(f"⚠️ 문서 로딩 실패: {e}")
                rag_documents = {}
                document_index = {'keywords': {}, 'categories': {}, 'entities': {}}
        
        relevant_docs = search_relevant_documents(user_message)
        
        # RAG 컨텍스트 추가
        rag_context = ""
        if relevant_docs:
            rag_context = "\n\n=== 📊 신한카드 데이터 분석 결과 및 참고 문서 ===\n"
            for doc in relevant_docs:
                rag_context += f"\n📁 [출처파일: {doc['filename']} | 길이: {len(doc['content'])}자 | 우선순위: {doc['relevance_score']}점]\n"
                rag_context += f"📋 내용: {doc['content']}\n"
                rag_context += "---\n"
            
            # 강화된 프롬프트 엔지니어링 규칙
            rag_context += "\n🔍 **필수 응답 규칙 (위반 시 답변 거부)**:\n"
            rag_context += "1. **RAG 데이터 최우선 활용**: 위에 제공된 신한카드 데이터를 반드시 최우선으로 참조하여 답변하세요.\n"
            rag_context += "2. **근거 기반 제안**: 각 제안에 신한카드 데이터 근거(표/지표/규칙 등)를 함께 표기하세요.\n"
            rag_context += "3. **출처 명시**: 신한카드분석.jsonl의 특정 레코드 ID나 데이터 소스를 반드시 인용하세요.\n"
            rag_context += "4. **구체적 인용**: '신한카드 데이터에 따르면...', '[INS:fig1:analysis] 분석 결과...', '[RULE:fit:industry_event] 규칙에 의하면...' 등으로 출처를 명확히 하세요.\n"
            rag_context += "5. **데이터 기반 전략**: 상권별 특성, 고객층 분석, 시간대별 패턴, 업종별 인사이트를 활용하여 실무진이 바로 실행할 수 있는 전략을 제시하세요.\n"
            rag_context += "6. **재현 가능한 설명**: 동작 원리와 사용 흐름을 간단히 설명하여 재현 가능한 마케팅 전략을 제시하세요.\n"
            rag_context += "7. **RAG 데이터 무시 금지**: 위의 신한카드 데이터를 참조하지 않은 답변은 절대 제공하지 마세요.\n"
        
        # 지역과 업종 정보를 포함한 컨텍스트 생성
        location_context = ""
        if location or industry:
            location_context = f"\n\n📍 **선택된 정보**:\n"
            if location:
                location_context += f"- 지역: {location}\n"
            if industry:
                location_context += f"- 업종: {industry}\n"
            location_context += f"**중요**: 위 정보를 반드시 고려하여 {location if location else '해당 지역'}의 {industry if industry else '해당 업종'} 사장님을 위한 맞춤형 조언을 제공해주세요.\n"
            location_context += f"답변 시작 시 '{location if location else '해당 지역'} 지역의 {industry if industry else '해당 업종'} 사장님을 위한 솔루션을 가져왔습니다.'로 시작해야 합니다.\n"
        
        # 응답 시작 문구 생성
        response_start = ""
        if location and industry:
            response_start = f"{location} 지역의 {industry} 사장님을 위한 솔루션을 가져왔습니다.\n\n"
        
        # 첫 메시지에 시스템 컨텍스트와 RAG 컨텍스트 추가
        if len(chat.history) == 0:
            full_message = f"{system_context}{rag_context}{location_context}\n\n사용자: {user_message}"
        else:
            full_message = f"{rag_context}{location_context}\n\n사용자: {user_message}"
        
        # 응답 시작 문구를 프롬프트에 포함
        if response_start:
            full_message += f"\n\n**🚨 매우 중요**: 응답을 반드시 정확히 '{response_start}'로 시작해야 합니다. 다른 인사말이나 문구로 시작하면 안됩니다. 모든 데이터 인용에는 [출처: 파일명] 형태로 출처를 표기해야 합니다. 위의 신한카드 데이터를 반드시 참조하여 답변하세요."
        
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
    return jsonify({'status': 'ok'}), 200

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

