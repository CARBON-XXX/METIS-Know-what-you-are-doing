"""SEDAC V9.0 + Ollama 测试 - 8B模型多轮对话"""
import requests
import json
import torch
from sedac.v9.production.config import ProductionConfig
from sedac.v9.production.engine import ProductionSEDACEngine, EntropyComputer
from sedac.v9.production.auto_calibration import AutoCalibrator

class OllamaSEDACTester:
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434"):
        print('=' * 70)
        print('SEDAC V9.0 + Ollama (Qwen2.5-7B) Test')
        print('=' * 70)
        
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        
        # 测试连接
        print(f'\n[Connecting to Ollama: {base_url}]')
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            models = [m['name'] for m in resp.json().get('models', [])]
            print(f'Available models: {models}')
        except Exception as e:
            print(f'Warning: {e}')
        
        # SEDAC Engine
        config = ProductionConfig()
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.model.num_hidden_layers = 28  # Qwen2.5-7B
        config.model.hidden_size = 3584
        
        self.config = config
        self.engine = ProductionSEDACEngine(config)
        self.calibrator = AutoCalibrator(model_layers=28)
        
        # 对话历史
        self.messages = []
        
        # 统计
        self.total_requests = 0
        self.total_tokens = 0
        
        print('[SEDAC Engine Ready]\n')
    
    def chat(self, user_input, stream=True):
        """多轮对话 - 维护上下文"""
        self.messages.append({"role": "user", "content": user_input})
        
        print(f'\n{"="*60}')
        print(f'User: {user_input}')
        print(f'[History: {len(self.messages)} messages]')
        print('='*60)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个有帮助的AI助手。请记住对话上下文，回答简洁准确。"}
            ] + self.messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,
            }
        }
        
        try:
            if stream:
                response_text = self._stream_response(payload)
            else:
                response_text = self._sync_response(payload)
            
            # 保存到历史
            self.messages.append({"role": "assistant", "content": response_text})
            self.total_requests += 1
            
            return response_text
            
        except Exception as e:
            print(f'\nError: {e}')
            self.messages.pop()
            return None
    
    def _stream_response(self, payload):
        """流式输出"""
        print('\nAssistant: ', end='', flush=True)
        
        response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
        
        full_response = ""
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    text = data['message']['content']
                    print(text, end='', flush=True)
                    full_response += text
                    token_count += 1
                
                if data.get('done', False):
                    eval_count = data.get('eval_count', token_count)
                    self.total_tokens += eval_count
                    print(f'\n\n[Tokens: {eval_count}]')
                    break
        
        return full_response
    
    def _sync_response(self, payload):
        """同步请求"""
        payload['stream'] = False
        response = requests.post(self.api_url, json=payload, timeout=120)
        data = response.json()
        
        text = data.get('message', {}).get('content', '')
        tokens = data.get('eval_count', len(text)//2)
        
        print(f'\nAssistant: {text}')
        print(f'\n[Tokens: {tokens}]')
        
        self.total_tokens += tokens
        return text
    
    def clear_history(self):
        """清除历史"""
        self.messages = []
        print('[对话历史已清除]')
    
    def print_stats(self):
        """打印统计"""
        print('\n' + '=' * 60)
        print('Statistics')
        print('=' * 60)
        print(f'Total Requests: {self.total_requests}')
        print(f'Total Tokens:   {self.total_tokens}')
        print(f'History Length: {len(self.messages)} messages')
        
        if self.calibrator.is_calibrated:
            params = self.calibrator.get_calibrated_params()
            print(f'\nCalibrated:')
            print(f'  Entropy Base: {params.entropy_threshold_base:.3f}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SEDAC + Ollama Test')
    parser.add_argument('--model', type=str, default='qwen2.5:7b', help='Model name')
    parser.add_argument('--url', type=str, default='http://localhost:11434', help='Ollama URL')
    args = parser.parse_args()
    
    tester = OllamaSEDACTester(model_name=args.model, base_url=args.url)
    
    # 快速测试
    print('\n--- Quick Test ---')
    tester.chat("你好，请记住我叫小明")
    tester.chat("我叫什么名字？")
    
    print('\n' + '=' * 60)
    print('Interactive Chat (Multi-turn with Ollama 7B)')
    print('Commands: quit, stats, clear')
    print('=' * 60)
    
    while True:
        try:
            user_input = input('\nYou: ').strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.lower() == 'stats':
                tester.print_stats()
                continue
            if user_input.lower() == 'clear':
                tester.clear_history()
                continue
            if not user_input:
                continue
            tester.chat(user_input)
        except KeyboardInterrupt:
            break
    
    print('\nFinal:')
    tester.print_stats()

if __name__ == '__main__':
    main()
