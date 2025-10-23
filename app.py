from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import pandas as pd
import json
import csv
import concurrent.futures
import random
import time

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# ====== 전역 변수 ======
user_setups = {}  # 사용자 설정 저장
response_cache = {}  # 응답 캐시

# ====== 상수 ======
GEN_MAX_OUTPUT_TOKENS = 4096
GEN_TIMEOUT_SEC = 180
MD_TIMEOUT_SEC = 10
RETRY_MAX = 3
RETRY_BASE = 2
RETRY_JITTER = 1

# ====== 유틸리티 함수 ======
def safe_markdown(text):
    """안전한 마크다운 변환"""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            f = ex.submit(markdown.markdown, text, extensions=['extra'])
            try:
                return f.result(timeout=MD_TIMEOUT_SEC)
            except Exception:
                return text
    except Exception:
        return text

def is_overloaded_error(msg):
    """오버로드 오류 감지"""
    overloaded_keywords = ['overloaded', 'quota', 'limit', 'rate']
    return any(keyword in msg.lower() for keyword in overloaded_keywords)

def get_model():
    """모델 가져오기"""
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
        return None
    else:
        print("✅ Google API Key가 설정되었습니다.")
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("🤖 Gemini 모델 로드 완료")
        return model

def load_calendar_events():
    """
    달력 이벤트 로드(두 CSV 통합)
    - 컬럼명은 질문에 올려주신 헤더와 동일하게 사용
    - 날짜 'YYYY.M.D' → ISO8601
    - end는 exclusive로 +1일
    - 인코딩 자동 판별
    - 중복 Event_ID 제거
    """
    from datetime import datetime, timedelta
    
    def _read_csv_any(path):
        """UTF-8(BOM) → CP949 → EUC-KR 순으로 시도"""
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(path, encoding=enc)
                return df
            except Exception:
                continue
        return None

    def _parse_dot_date(s: str) -> str:
        """
        '2023.1.9' → '2023-01-09'
        공백/None/빈값 안전 처리
        """
        if not s or not str(s).strip():
            return ""
        s = str(s).strip().rstrip(".")            # '2023.1.9.' 같은 꼬리 점 제거
        try:
            dt = datetime.strptime(s, "%Y.%m.%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # 혹시 '2023.01.09' 처럼 0패딩이 있어도 위 포맷이 처리하니 여기로 잘 안옴.
            # 예외 시 원문 반환(디버깅을 위해)
            return s

    def _norm_row(row: dict, src: str, idx: int) -> dict:
        """CSV 한 행을 이벤트 객체로 정규화"""
        start = _parse_dot_date(row.get("Start_Date", ""))
        end_inclusive = _parse_dot_date(row.get("End_Date", ""))

        # FullCalendar 등은 end가 exclusive이므로, 종료일이 있으면 하루 +1
        end_exclusive = ""
        if end_inclusive:
            try:
                end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                end_exclusive = end_inclusive  # 실패 시라도 그대로

        eid = row.get("Event_ID") or f"{src}_{idx}"
        title = row.get("Event_Name", "").strip()
        evtype = row.get("Event_Type", "").strip()
        location = row.get("Location_Address", "") or ""
        audience = row.get("Target_Audience", "") or ""
        district = row.get("Associated_District", "") or ""
        desc = row.get("Event_Description", "") or ""

        return {
            "id": str(eid),
            "title": title,
            "start": start,
            "end": end_exclusive,      # ← exclusive로 전달
            "allDay": True,            # 날짜 단위 이벤트 가정
            "type": evtype,
            "location": location,
            "audience": audience,
            "district": district,
            "description": desc,
            "source": src,
        }

    events = []
    seen_ids = set()

    files = [
        ("공통이벤트", os.path.join(app.root_path, "documents", "raw", "성동구 공통_한양대_흥행영화 이벤트 DB.csv")),
        ("성수팝업", os.path.join(app.root_path, "documents", "raw", "성수 팝업 최종.csv")),
    ]

    for label, path in files:
        if not os.path.exists(path):
            continue
        df = _read_csv_any(path)
        if df is None:
            print(f"[캘린더] CSV 인코딩 실패: {path}")
            continue

        # 열 이름 공백 제거/표준화(엑셀 저장 시 공백이 끼는 경우 대비)
        df.columns = [str(c).strip() for c in df.columns]

        for i, row in df.iterrows():
            try:
                ev = _norm_row(row, label, i)
                # start가 비면 스킵
                if not ev["start"]:
                    continue
                # 중복 제거(Event_ID 기준)
                if ev["id"] in seen_ids:
                    continue
                seen_ids.add(ev["id"])
                events.append(ev)
            except Exception as e:
                print(f"[캘린더] 행 파싱 실패 {label}#{i}: {e}")

    print(f"📅 총 {len(events)}개 이벤트 로드 완료")
    return events

def load_shinhan_data():
    """신한카드 분석 데이터 로드"""
    shinhan_data = []
    jsonl_path = os.path.join(app.root_path, 'documents', 'raw', '신한카드분석.jsonl')
    
    try:
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        shinhan_data.append(data)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error loading Shinhan data: {e}")
    
    return shinhan_data

def search_relevant_documents(query, location="", industry=""):
    """관련 문서 검색"""
    shinhan_data = load_shinhan_data()
    relevant_snippets = []
    
    # 신한카드 데이터에서 관련 스니펫 검색
    for item in shinhan_data:
        content = str(item.get('content', ''))
        if query.lower() in content.lower():
            relevant_snippets.append({
                'content': content[:500],  # 첫 500자만
                'source': '신한카드분석.jsonl',
                'priority': 1
            })
    
    # 지역/업종 매칭
    if location or industry:
        for item in shinhan_data:
            content = str(item.get('content', ''))
            if (location and location in content) or (industry and industry in content):
                relevant_snippets.append({
                    'content': content[:500],
                    'source': '신한카드분석.jsonl',
                    'priority': 1
                })
    
    return relevant_snippets[:3]  # 상위 3개만 반환

def call_gemini_with_retry(model, prompt: str):
    """안정화된 Gemini API 호출"""
    generation_config = {
        "max_output_tokens": GEN_MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    last_err = None

    for attempt in range(RETRY_MAX):
        try:
            print(f"🔄 Gemini API 호출 시도 {attempt+1}/{RETRY_MAX}")
            print(f"📝 프롬프트 길이: {len(prompt)}자")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                )
                resp = fut.result(timeout=GEN_TIMEOUT_SEC)

            print(f"✅ Gemini API 응답 받음: type={type(resp)}")
            
            # 응답 텍스트 추출
            try:
                text_output = resp.text
            except Exception as e:
                print(f"⚠️ 응답 텍스트 추출 실패: {e}")
                text_output = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 질문을 다시 간단히 해주시거나 다른 방식으로 문의해 주세요."
            
            if not text_output or len(text_output.strip()) < 10:
                text_output = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 질문을 다시 간단히 해주시거나 다른 방식으로 문의해 주세요."
                print("⚠️ 빈 응답으로 인한 안전 문구 사용")
            
            print(f"📤 최종 응답 길이: {len(text_output)}자")
            return text_output

        except concurrent.futures.TimeoutError:
            last_err = TimeoutError(f"generate_content timeout ({GEN_TIMEOUT_SEC}s)")
        except Exception as e:
            last_err = e
            msg = str(e)
            if is_overloaded_error(msg) and attempt < RETRY_MAX - 1:
                backoff = (RETRY_BASE ** attempt) + random.uniform(0, RETRY_JITTER)
                print(f"⚠️ 일시 오류(재시도 {attempt+1}/{RETRY_MAX}): {msg} → {backoff:.2f}s 대기")
                time.sleep(backoff)
            else:
                print(f"❌ API Error: {msg}")
                break

    # 최종 실패 시 에러 메시지 반환
    error_msg = f"API 호출 실패: {str(last_err)}"
    print(f"❌ {error_msg}")
    return f"죄송합니다. 응답 생성 중 문제가 발생했습니다. ({error_msg})"

# ====== 라우트 ======
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/setup')
def setup():
    """설정 페이지"""
    return render_template('setup.html')

@app.route('/chat')
def chat():
    """채팅 페이지"""
    return render_template('chat.html')

@app.route('/api/check-setup', methods=['GET'])
def check_setup():
    """설정 확인"""
    setup_exists = 'default' in user_setups and user_setups['default']
    setup_info = user_setups.get('default', {}) if setup_exists else None
    
    return jsonify({
        'setup_exists': setup_exists,
        'is_setup': setup_exists,
        'setup_info': setup_info
    })

@app.route('/api/setup', methods=['POST'])
def save_setup():
    """사용자 설정 저장"""
    try:
        data = request.get_json()
        location = data.get('location', '')
        industry = data.get('industry', '')
        store_name = data.get('store_name', '')
        
        user_setups['default'] = {
            'location': location,
            'industry': industry,
            'store_name': store_name
        }
        
        print(f"📍 지역: {location}, 🏢 업종: {industry}, 🏪 가게: {store_name}")
        
        return jsonify({'success': True, 'message': '설정이 저장되었습니다.'})
    
    except Exception as e:
        print(f"Error saving setup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-setup', methods=['GET'])
def get_setup():
    """사용자 설정 조회"""
    setup = user_setups.get('default', {})
    return jsonify(setup)

@app.route('/api/calendar-events', methods=['GET'])
def get_calendar_events():
    """달력 이벤트 조회"""
    try:
        events = load_calendar_events()
        return jsonify(events)
    except Exception as e:
        print(f"Error loading calendar events: {e}")
        return jsonify([])

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """채팅 API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        # 사용자 설정 가져오기
        setup = user_setups.get('default', {})
        location = setup.get('location', '')
        industry = setup.get('industry', '')
        store_name = setup.get('store_name', '')
        
        # 캐시 확인
        cache_key = f"{user_message}_{location}_{industry}"
        if cache_key in response_cache:
            print("💾 캐시에서 응답 반환")
            return jsonify({
                'message': response_cache[cache_key],
                'session_id': session_id
            })
        
        # Gemini API 호출
        model = get_model()
        if model is None:
            return jsonify({
                'message': '❌ Google API Key가 설정되지 않았습니다.',
                'status': 'error'
            }), 500
        
        # RAG 검색
        relevant_docs = search_relevant_documents(user_message, location, industry)
        
        # 프롬프트 구성
        context_info = ""
        if location or industry or store_name:
            context_info = f"""
📍 선택된 지역: {location if location else '미선택'}
🏪 선택된 업종: {industry if industry else '미선택'}
🏢 가게명: {store_name if store_name else '미선택'}

위의 지역과 업종 정보를 고려하여 답변해주세요.
"""
        
        # RAG 컨텍스트 추가
        rag_context = ""
        if relevant_docs:
            rag_context = "\n\n[참고 데이터]\n"
            for i, doc in enumerate(relevant_docs, 1):
                rag_context += f"{i}. {doc['content']}\n"
        
        prompt = f"""
안녕하세요! 성동구 소상공인 여러분을 위한 마케팅 도우미입니다.

{context_info}
{rag_context}

질문: {user_message}

성동구 지역의 소상공인에게 도움이 되는 실용적인 마케팅 조언을 제공해주세요.
특히 선택된 지역과 업종에 맞는 구체적이고 실용적인 조언을 해주세요.
참고 데이터가 있다면 적극적으로 활용해주세요.
"""
        
        # Gemini API 호출
        start_time = time.time()
        response_text = call_gemini_with_retry(model, prompt)
        
        # 마크다운 변환
        html_response = safe_markdown(response_text)
        
        # 캐시 저장
        response_cache[cache_key] = html_response
        
        # 처리 시간 로깅
        processing_time = time.time() - start_time
        print(f"⏱️ API 응답 시간: {processing_time:.2f}초")
        
        return jsonify({
            'message': html_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error in chat API: {e}")
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """채팅 리셋"""
    global response_cache
    response_cache.clear()
    return jsonify({'success': True, 'message': '채팅이 리셋되었습니다.'})

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Flask 서버 시작: 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)