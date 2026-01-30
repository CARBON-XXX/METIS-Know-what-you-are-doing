"""SEDAC V9.0 + vLLM 测试 - 连接WSL中的8B模型"""
import os
import torch
import requests
import json
from openai import OpenAI
from sedac.v9.production.config import ProductionConfig
from sedac.v9.production.engine import ProductionSEDACEngine, EntropyComputer
from sedac.v9.production.auto_calibration import AutoCalibrator

class VLLMSEDACTester:
    def __init__(self, vllm_url="http://localhost:8000/v1", model_name="Qwen/Qwen2.5-7B-Instruct"):
        print('=' * 70)
        print('SEDAC V9.0 + vLLM (8B Model) Test')
        print('=' * 70)
        
        self.vllm_url = vllm_url
        self.model_name = model_name
        
        # OpenAI兼容客户端
        self.client = OpenAI(
            base_url=vllm_url,
            api_key="EMPTY"  # vLLM不需要真实key
        )
        
        # 测试连接
        print(f'\n[Connecting to vLLM: {vllm_url}]')
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            print(f'Available models: {available}')
            if available:
                self.model_name = available[0]
                print(f'Using: {self.model_name}')
        except Exception as e:
            print(f'Warning: Could not list models: {e}')
        
        # SEDAC Engine (用于熵计算演示)
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
    
    def chat(self, user_input, max_tokens=256, stream=True):
        """多轮对话"""
        self.messages.append({"role": "user", "content": user_input})
        
        print(f'\n{"="*60}')
        print(f'User: {user_input}')
        print(f'[History: {len(self.messages)} messages]')
        print('='*60)
        
        try:
            if stream:
                response_text = self._stream_chat(max_tokens)
            else:
                response_text = self._sync_chat(max_tokens)
            
            # 保存到历史
            self.messages.append({"role": "assistant", "content": response_text})
            self.total_requests += 1
            
            return response_text
            
        except Exception as e:
            print(f'\nError: {e}')
            # 移除失败的用户消息
            self.messages.pop()
            return None
    
    def _stream_chat(self, max_tokens):
        """流式输出"""
        print('\nAssistant: ', end='', flush=True)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手。请记住对话上下文。"},
            ] + self.messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )
        
        full_response = ""
        token_count = 0
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end='', flush=True)
                full_response += text
                token_count += 1
        
        print()  # 换行
        self.total_tokens += token_count
        print(f'\n[Tokens: {token_count}]')
        
        return full_response
    
    def _sync_chat(self, max_tokens):
        """同步请求"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手。请记住对话上下文。"},
            ] + self.messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        text = response.choices[0].message.content
        tokens = response.usage.completion_tokens if response.usage else len(text)//2
        
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SEDAC + vLLM Test')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1',
                       help='vLLM server URL')
    parser.add_argument('--model', type=str, default='',
                       help='Model name (auto-detect if empty)')
    args = parser.parse_args()
    
    tester = VLLMSEDACTester(vllm_url=args.url, model_name=args.model)
    
    print('\n' + '=' * 60)
    print('Interactive Chat (Multi-turn with vLLM 8B)')
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
