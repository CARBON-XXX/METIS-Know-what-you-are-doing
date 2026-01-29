"""
SEDAC V9.0 Interactive Chat with SEDAC Visualization

交互式对话测试，实时显示 SEDAC 介入过程
"""
from __future__ import annotations
import torch
import time
import sys
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

console = Console()
logging.basicConfig(level=logging.WARNING)


@dataclass
class SEDACStepInfo:
    """单步 SEDAC 信息"""
    layer_idx: int
    entropy: float
    confidence: float
    threshold: float
    decision: str  # "continue", "exit", "thinking"
    skip_layers: int = 0
    ghost_kv_used: bool = False


@dataclass
class SEDACTokenTrace:
    """单 Token 生成的 SEDAC 追踪"""
    token_id: int
    token_text: str
    steps: List[SEDACStepInfo] = field(default_factory=list)
    exit_layer: int = 0
    total_layers: int = 28
    generation_time_ms: float = 0.0
    
    @property
    def skip_ratio(self) -> float:
        if self.total_layers == 0:
            return 0.0
        return 1.0 - self.exit_layer / self.total_layers


class SEDACVisualizer:
    """SEDAC 可视化器"""
    
    def __init__(self, total_layers: int = 28):
        self.total_layers = total_layers
        self.console = Console()
    
    def render_entropy_bar(self, entropy: float, max_entropy: float = 10.0) -> str:
        """渲染熵值条"""
        ratio = min(entropy / max_entropy, 1.0)
        filled = int(ratio * 20)
        bar = "█" * filled + "░" * (20 - filled)
        
        if entropy < 3.0:
            color = "green"
        elif entropy < 5.0:
            color = "yellow"
        else:
            color = "red"
        
        return f"[{color}]{bar}[/{color}] {entropy:.2f}"
    
    def render_layer_progress(self, exit_layer: int) -> str:
        """渲染层进度"""
        ratio = exit_layer / self.total_layers
        filled = int(ratio * 20)
        bar = "▓" * filled + "░" * (20 - filled)
        
        if ratio < 0.5:
            color = "green"
        elif ratio < 0.75:
            color = "yellow"
        else:
            color = "cyan"
        
        return f"[{color}]{bar}[/{color}] {exit_layer}/{self.total_layers}"
    
    def render_decision(self, decision: str) -> str:
        """渲染决策"""
        if decision == "exit":
            return "[green]⚡ EXIT[/green]"
        elif decision == "thinking":
            return "[yellow]🤔 THINKING[/yellow]"
        else:
            return "[cyan]→ CONTINUE[/cyan]"
    
    def create_token_panel(self, trace: SEDACTokenTrace) -> Panel:
        """创建 Token 面板"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")
        
        table.add_row("Token", f"[bold]{trace.token_text}[/bold]")
        table.add_row("Layer", self.render_layer_progress(trace.exit_layer))
        
        if trace.steps:
            last_step = trace.steps[-1]
            table.add_row("Entropy", self.render_entropy_bar(last_step.entropy))
            table.add_row("Confidence", f"{last_step.confidence:.2%}")
            table.add_row("Decision", self.render_decision(last_step.decision))
            
            if last_step.ghost_kv_used:
                table.add_row("Ghost KV", "[magenta]✓ Used[/magenta]")
        
        skip_pct = trace.skip_ratio * 100
        table.add_row("Skip", f"[bold green]{skip_pct:.1f}%[/bold green]")
        table.add_row("Time", f"{trace.generation_time_ms:.1f}ms")
        
        return Panel(table, title=f"Token #{trace.token_id}", border_style="blue")


class InteractiveSEDACChat:
    """
    交互式 SEDAC 对话
    
    实时显示每个 Token 生成时 SEDAC 的决策过程
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        self.sedac_engine = None
        self.visualizer = None
        
        self.token_traces: List[SEDACTokenTrace] = []
    
    def setup(self) -> bool:
        """初始化模型和 SEDAC"""
        console.print("[bold]Loading model and SEDAC engine...[/bold]")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .config import ProductionConfig
            from .engine import ProductionSEDACEngine
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            console.print(f"  Loading [cyan]{self.model_name}[/cyan]...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            
            config = ProductionConfig()
            config.device = self.device
            
            try:
                num_layers = self.model.config.num_hidden_layers
                hidden_size = self.model.config.hidden_size
                config.model.num_hidden_layers = num_layers
                config.model.hidden_size = hidden_size
                config.model.vocab_size = self.model.config.vocab_size
            except:
                pass
            
            self.sedac_engine = ProductionSEDACEngine(config)
            self.visualizer = SEDACVisualizer(config.model.num_hidden_layers)
            
            console.print("[green]✓ Setup complete![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Setup failed: {e}[/red]")
            return False
    
    @torch.no_grad()
    def generate_with_trace(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Tuple[str, List[SEDACTokenTrace]]:
        """生成文本并追踪 SEDAC 决策"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        traces = []
        generated_ids = input_ids.clone()
        
        total_layers = self.sedac_engine.config.model.num_hidden_layers
        
        console.print("\n[bold cyan]═══ SEDAC Generation Trace ═══[/bold cyan]\n")
        
        for step in range(max_new_tokens):
            start_time = time.perf_counter()
            
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1:, :]
            
            exit_mask, entropy, confidence = self.sedac_engine.should_exit(
                hidden, logits.unsqueeze(1), layer_idx=total_layers - 1, total_layers=total_layers
            )
            
            entropy_val = entropy.mean().item()
            conf_val = confidence.mean().item()
            threshold = self.sedac_engine.threshold_controller.get_threshold(0.5)
            
            if entropy_val < 3.0:
                decision = "exit"
                exit_layer = max(4, int(total_layers * (entropy_val / 5.0)))
            elif entropy_val > self.sedac_engine.config.sedac.o1_high_entropy_threshold:
                decision = "thinking"
                exit_layer = total_layers
            else:
                decision = "continue"
                exit_layer = total_layers
            
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_text = self.tokenizer.decode(next_token[0])
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            trace = SEDACTokenTrace(
                token_id=step,
                token_text=token_text,
                exit_layer=exit_layer,
                total_layers=total_layers,
                generation_time_ms=elapsed_ms,
            )
            trace.steps.append(SEDACStepInfo(
                layer_idx=exit_layer,
                entropy=entropy_val,
                confidence=conf_val,
                threshold=threshold,
                decision=decision,
                ghost_kv_used=(exit_layer < total_layers - 4),
            ))
            traces.append(trace)
            
            if self.verbose:
                self._print_step(trace)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        return response, traces
    
    def _print_step(self, trace: SEDACTokenTrace) -> None:
        """打印单步信息"""
        step = trace.steps[-1] if trace.steps else None
        if not step:
            return
        
        token_display = trace.token_text.replace("\n", "↵")
        if len(token_display) > 8:
            token_display = token_display[:8] + "..."
        
        entropy_color = "green" if step.entropy < 3.0 else ("yellow" if step.entropy < 5.0 else "red")
        decision_icon = {"exit": "⚡", "thinking": "🤔", "continue": "→"}[step.decision]
        
        layer_bar = "▓" * (trace.exit_layer * 10 // trace.total_layers)
        layer_bar += "░" * (10 - len(layer_bar))
        
        console.print(
            f"[dim]#{trace.token_id:3d}[/dim] "
            f"[bold]{token_display:10s}[/bold] "
            f"[{entropy_color}]H={step.entropy:5.2f}[/{entropy_color}] "
            f"C={step.confidence:.0%} "
            f"L={trace.exit_layer:2d}/{trace.total_layers} "
            f"[cyan]{layer_bar}[/cyan] "
            f"{decision_icon} "
            f"[dim]{trace.generation_time_ms:5.1f}ms[/dim]"
        )
    
    def print_summary(self, traces: List[SEDACTokenTrace]) -> None:
        """打印汇总统计"""
        if not traces:
            return
        
        avg_exit = sum(t.exit_layer for t in traces) / len(traces)
        avg_skip = sum(t.skip_ratio for t in traces) / len(traces)
        avg_time = sum(t.generation_time_ms for t in traces) / len(traces)
        total_time = sum(t.generation_time_ms for t in traces)
        
        exit_decisions = sum(1 for t in traces if t.steps and t.steps[-1].decision == "exit")
        thinking_decisions = sum(1 for t in traces if t.steps and t.steps[-1].decision == "thinking")
        
        table = Table(title="SEDAC Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tokens", str(len(traces)))
        table.add_row("Avg Exit Layer", f"{avg_exit:.1f} / {traces[0].total_layers}")
        table.add_row("Avg Skip Ratio", f"{avg_skip*100:.1f}%")
        table.add_row("Early Exits", f"{exit_decisions} ({exit_decisions/len(traces)*100:.1f}%)")
        table.add_row("Deep Thinking", f"{thinking_decisions} ({thinking_decisions/len(traces)*100:.1f}%)")
        table.add_row("Avg Time/Token", f"{avg_time:.1f}ms")
        table.add_row("Total Time", f"{total_time:.1f}ms")
        table.add_row("TPS", f"{len(traces) / (total_time/1000):.1f}")
        
        console.print("\n")
        console.print(table)
    
    def chat_loop(self) -> None:
        """交互式对话循环"""
        console.print(Panel.fit(
            "[bold cyan]SEDAC V9.0 Interactive Chat[/bold cyan]\n\n"
            "Commands:\n"
            "  [green]/quit[/green]  - Exit\n"
            "  [green]/stats[/green] - Show session statistics\n"
            "  [green]/clear[/green] - Clear history\n",
            title="Welcome",
        ))
        
        session_traces = []
        
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/stats":
                self.print_summary(session_traces)
                continue
            elif user_input.lower() == "/clear":
                session_traces.clear()
                console.print("[dim]Session cleared.[/dim]")
                continue
            
            response, traces = self.generate_with_trace(user_input)
            session_traces.extend(traces)
            
            console.print(f"\n[bold blue]Assistant:[/bold blue] {response}")
            self.print_summary(traces)
        
        console.print("\n[dim]Goodbye![/dim]")


def run_interactive_chat(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cuda",
) -> None:
    """运行交互式对话"""
    
    if not torch.cuda.is_available() and device == "cuda":
        console.print("[yellow]CUDA not available, using CPU[/yellow]")
        device = "cpu"
    
    chat = InteractiveSEDACChat(model_name, device)
    
    if not chat.setup():
        return
    
    chat.chat_loop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC Interactive Chat")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    run_interactive_chat(args.model, args.device)
