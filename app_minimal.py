from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini API 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_model():
    """모델 가져오기"""
    if not GOOGLE_API_KEY:
        print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
        return None
    else:
        print("✅ Google API Key가 설정되었습니다.")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("🤖 Gemini 모델 로드 완료")
        return model

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """채팅 API - 최소한의 기능만"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        # Gemini API 호출
        model = get_model()
        if model is None:
            return jsonify({
                'message': '❌ Google API Key가 설정되지 않았습니다.',
                'status': 'error'
            }), 500
        
        # 지역과 업종 정보 가져오기
        location = data.get('location', '')
        industry = data.get('industry', '')
        
        # 지역과 업종 정보를 포함한 프롬프트 생성
        context_info = ""
        if location or industry:
            context_info = f"""
📍 선택된 지역: {location if location else '미선택'}
🏪 선택된 업종: {industry if industry else '미선택'}

위의 지역과 업종 정보를 고려하여 답변해주세요.
"""
        
        # 향상된 프롬프트
        prompt = f"""
        안녕하세요! 성동구 소상공인 여러분을 위한 마케팅 도우미입니다.
        
        {context_info}
        
        질문: {user_message}
        
        성동구 지역의 소상공인에게 도움이 되는 실용적인 마케팅 조언을 제공해주세요.
        특히 선택된 지역과 업종에 맞는 구체적이고 실용적인 조언을 해주세요.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        html_response = markdown.markdown(response_text, extensions=['extra'])
        
        return jsonify({
            'message': html_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Flask 서버 시작: 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
