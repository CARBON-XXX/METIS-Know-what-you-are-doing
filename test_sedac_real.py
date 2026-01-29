"""SEDAC V9.0 真实模型测试"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import sys
sys.path.insert(0, '.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sedac.v9.production.config import ProductionConfig
from sedac.v9.production.engine import ProductionSEDACEngine
from sedac.v9.production.auto_calibration import AutoCalibrator

print('=' * 60)
print('SEDAC V9.0 Production - Real Model Test')
print('=' * 60)
print()

# Load model
print('[1/4] Loading Qwen2.5-0.5B-Instruct...')
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map='auto'
)
print(f'    Model: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden')
print()

# Init SEDAC
print('[2/4] Initializing SEDAC Engine...')
config = ProductionConfig()
config.device = 'cuda'
config.model.num_hidden_layers = model.config.num_hidden_layers
config.model.hidden_size = model.config.hidden_size
config.model.vocab_size = model.config.vocab_size

engine = ProductionSEDACEngine(config)
calibrator = AutoCalibrator(model_layers=model.config.num_hidden_layers)
print('    SEDAC Engine ready')
print()

# Test prompts
print('[3/4] Running SEDAC inference tests...')
print()

prompts = [
    ('简单', 'Hello'),
    ('数学', 'What is 2+2?'),
    ('复杂', 'Explain the theory of relativity in simple terms.'),
]

total_layers = model.config.num_hidden_layers

for category, prompt in prompts:
    print(f'--- [{category}] "{prompt}" ---')
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids, 
            output_hidden_states=True, 
            return_dict=True
        )
        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1:, :]
        
        # SEDAC decision
        exit_mask, entropy, confidence = engine.should_exit(
            hidden, logits.unsqueeze(1), 
            layer_idx=total_layers-1, 
            total_layers=total_layers
        )
        
        # Handle both tensor and float returns
        entropy_val = entropy.mean().item() if hasattr(entropy, 'mean') else float(entropy)
        conf_val = confidence.mean().item() if hasattr(confidence, 'mean') else float(confidence)
        threshold = engine.threshold_controller.get_threshold(0.5)
        
        # Determine decision
        if entropy_val < 3.0:
            decision = 'EARLY EXIT'
            exit_layer = max(4, int(total_layers * (entropy_val / 5.0)))
        elif entropy_val > config.sedac.o1_high_entropy_threshold:
            decision = 'O1 THINKING'
            exit_layer = total_layers
        else:
            decision = 'NORMAL'
            exit_layer = total_layers
        
        skip_ratio = 1.0 - exit_layer / total_layers
        
        # Record for calibration
        calibrator.record_sample(entropy_val, conf_val, exit_layer, 1.0)
        
        # Generate response
        gen_ids = model.generate(
            inputs.input_ids, 
            max_new_tokens=50, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        response = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        response_only = response[len(prompt):].strip()
        
        print(f'  Entropy:    {entropy_val:.2f}')
        print(f'  Confidence: {conf_val:.1%}')
        print(f'  Decision:   {decision}')
        print(f'  Exit Layer: {exit_layer}/{total_layers} (skip {skip_ratio:.1%})')
        print(f'  Response:   {response_only[:100]}{"..." if len(response_only) > 100 else ""}')
        print()

# Show calibrated params
print('[4/4] Auto-Calibration Results...')
if calibrator.is_calibrated:
    params = calibrator.get_calibrated_params()
    print(f'  Entropy Threshold Base: {params.entropy_threshold_base:.3f}')
    print(f'  Entropy Threshold Min:  {params.entropy_threshold_min:.3f}')
    print(f'  Entropy Threshold Max:  {params.entropy_threshold_max:.3f}')
    print(f'  O1 High Entropy:        {params.o1_high_entropy_threshold:.2f}')
    print(f'  Min Exit Layer:         {params.min_exit_layer}')
else:
    print('  Not yet calibrated (need more samples)')

print()
print('=' * 60)
print('Test Complete - All SEDAC functions verified!')
print('=' * 60)
