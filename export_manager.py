"""
Export dialogue data to various formats for research and analysis.
Supports CSV, Excel, JSON, and formatted reports.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import csv


class ExportManager:
    """Export dialogue data to multiple formats"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(
        self,
        history: List[Dict[str, Any]],
        filename: str = "dialogue_export.csv"
    ) -> Path:
        """Export dialogue to CSV format"""
        output_path = self.output_dir / filename
        
        if not history:
            return output_path
        
        # Define columns
        fieldnames = [
            "turn", "timestamp", "speaker", "speaker_nature",
            "target", "reply", "tone", "emotion", "word_count",
            "validation_issues", "env_context"
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in history:
                row = {
                    "turn": record.get("turn", 0),
                    "timestamp": record.get("ts_utc", ""),
                    "speaker": record.get("speaker", ""),
                    "speaker_nature": record.get("speaker_nature", ""),
                    "target": record.get("target", ""),
                    "reply": record.get("reply", ""),
                    "tone": record.get("tone", ""),
                    "emotion": record.get("emotion", ""),
                    "word_count": len(record.get("reply", "").split()),
                    "validation_issues": record.get("validation_issues", 0),
                    "env_context": record.get("env_context", "")
                }
                writer.writerow(row)
        
        return output_path
    
    def export_to_excel(
        self,
        history: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        filename: str = "dialogue_export.xlsx"
    ) -> Path:
        """Export dialogue to Excel with multiple sheets"""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            print("openpyxl not installed. Install with: pip install openpyxl")
            return self.output_dir / filename
        
        output_path = self.output_dir / filename
        wb = openpyxl.Workbook()
        
        # Sheet 1: Dialogue transcript
        ws_dialogue = wb.active
        ws_dialogue.title = "Dialogue"
        
        # Headers
        headers = [
            "Turn", "Timestamp", "Speaker", "Nature",
            "Target", "Reply", "Tone", "Emotion", "Words", "Issues"
        ]
        ws_dialogue.append(headers)
        
        # Style headers
        for cell in ws_dialogue[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
        
        # Data
        for record in history:
            row = [
                record.get("turn", 0),
                record.get("ts_utc", ""),
                record.get("speaker", ""),
                record.get("speaker_nature", ""),
                record.get("target", ""),
                record.get("reply", ""),
                record.get("tone", ""),
                record.get("emotion", ""),
                len(record.get("reply", "").split()),
                record.get("validation_issues", 0)
            ]
            ws_dialogue.append(row)
        
        # Auto-size columns
        for column in ws_dialogue.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws_dialogue.column_dimensions[column_letter].width = adjusted_width
        
        # Sheet 2: Metrics (if provided)
        if metrics:
            ws_metrics = wb.create_sheet("Metrics")
            ws_metrics.append(["Metric", "Window", "Total"])
            
            # Style header
            for cell in ws_metrics[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
                cell.font = Font(color="FFFFFF", bold=True)
            
            # Add metrics
            if "summary" in metrics:
                s_win = metrics["summary"].get("window", {})
                s_all = metrics["summary"].get("all", {})
                
                metric_rows = [
                    ("Messages", s_win.get("messages", 0), s_all.get("messages", 0)),
                    ("Avg Tone", f"{s_win.get('avg_tone', 0):.2f}", f"{s_all.get('avg_tone', 0):.2f}"),
                    ("Targeting Rate", f"{s_win.get('targeting_rate', 0):.2%}", f"{s_all.get('targeting_rate', 0):.2%}"),
                    ("Emotion Entropy", f"{s_win.get('emotion_entropy', 0):.2f}", f"{s_all.get('emotion_entropy', 0):.2f}"),
                    ("Avg Words", f"{s_win.get('avg_reply_words', 0):.1f}", f"{s_all.get('avg_reply_words', 0):.1f}"),
                    ("Reciprocity", f"{s_win.get('reciprocity', 0):.2f}", f"{s_all.get('reciprocity', 0):.2f}"),
                    ("Response Delay", f"{s_win.get('avg_addressing_delay', 0):.2f}", f"{s_all.get('avg_addressing_delay', 0):.2f}"),
                ]
                
                for row in metric_rows:
                    ws_metrics.append(row)
        
        wb.save(output_path)
        return output_path
    
    def export_to_json(
        self,
        history: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        filename: str = "dialogue_export.json"
    ) -> Path:
        """Export complete dataset to JSON"""
        output_path = self.output_dir / filename
        
        export_data = {
            "metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_turns": len(history),
                "env_context": history[0].get("env_context", "") if history else ""
            },
            "dialogue": history,
        }
        
        if metrics:
            export_data["metrics"] = metrics
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def export_research_report(
        self,
        history: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        filename: str = "research_report.md"
    ) -> Path:
        """Export formatted research report in Markdown"""
        output_path = self.output_dir / filename
        
        lines = []
        lines.append("# Dialogue Simulation Research Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Total Turns:** {len(history)}")
        
        if history:
            lines.append(f"**Environment:** {history[0].get('env_context', 'N/A')}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        if metrics and "summary" in metrics:
            s_all = metrics["summary"]["all"]
            
            avg_tone = s_all.get("avg_tone", 0)
            tone_desc = "positive" if avg_tone > 0.2 else ("negative" if avg_tone < -0.2 else "neutral")
            
            lines.append(f"- **Overall Tone:** {tone_desc} ({avg_tone:+.2f})")
            lines.append(f"- **Emotion Diversity:** {s_all.get('emotion_entropy', 0):.2f} (Shannon entropy)")
            lines.append(f"- **Reciprocity:** {s_all.get('reciprocity', 0):.2%}")
            lines.append(f"- **Average Message Length:** {s_all.get('avg_reply_words', 0):.1f} words")
            lines.append(f"- **Response Delay:** {s_all.get('avg_addressing_delay', 0):.2f} turns")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Detailed Metrics
        lines.append("## Detailed Metrics")
        lines.append("")
        
        if metrics and "summary" in metrics:
            s_all = metrics["summary"]["all"]
            
            lines.append("### Communication Patterns")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Total Messages | {s_all.get('messages', 0)} |")
            lines.append(f"| Targeting Rate | {s_all.get('targeting_rate', 0):.1%} |")
            lines.append(f"| Emotion Diversity (unique) | {s_all.get('emotion_diversity', 0)} |")
            lines.append("")
            
            lines.append("### Tone Distribution")
            lines.append("")
            tone_fracs = s_all.get("tone_fracs", {})
            lines.append("| Tone | Percentage |")
            lines.append("|------|------------|")
            lines.append(f"| Positive | {tone_fracs.get('pos', 0):.1%} |")
            lines.append(f"| Neutral | {tone_fracs.get('neu', 0):.1%} |")
            lines.append(f"| Negative | {tone_fracs.get('neg', 0):.1%} |")
            lines.append("")
        
        # Per-agent analysis
        if metrics and "per_agent" in metrics:
            lines.append("### Per-Agent Analysis")
            lines.append("")
            lines.append("| Agent | Messages | Avg Words | Avg Tone | Last Emotion | Target Diversity |")
            lines.append("|-------|----------|-----------|----------|--------------|------------------|")
            
            for agent, stats in metrics["per_agent"].items():
                lines.append(
                    f"| {agent} | {stats['msgs']} | {stats['avg_words']:.1f} | "
                    f"{stats['avg_tone']:+.2f} | {stats['last_emotion']} | {stats['targets_diversity']} |"
                )
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Sample dialogue
        lines.append("## Sample Dialogue Excerpt")
        lines.append("")
        
        sample_size = min(10, len(history))
        for record in history[:sample_size]:
            speaker = record.get("speaker", "Unknown")
            target = record.get("target", "all")
            reply = record.get("reply", "")
            tone = record.get("tone", "neutral")
            emotion = record.get("emotion", "neutral")
            
            lines.append(f"**Turn {record.get('turn', 0)}:** {speaker} → {target} ({tone}, {emotion})")
            lines.append(f"> {reply}")
            lines.append("")
        
        if len(history) > sample_size:
            lines.append(f"*... and {len(history) - sample_size} more turns*")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Agent Dialogue Simulator*")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def export_all_formats(
        self,
        history: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        base_name: str = "dialogue_export"
    ) -> Dict[str, Path]:
        """Export to all supported formats at once"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        exports = {}
        exports["csv"] = self.export_to_csv(history, f"{base_name}_{timestamp}.csv")
        exports["json"] = self.export_to_json(history, metrics, f"{base_name}_{timestamp}.json")
        
        if metrics:
            exports["excel"] = self.export_to_excel(history, metrics, f"{base_name}_{timestamp}.xlsx")
            exports["report"] = self.export_research_report(history, metrics, f"report_{timestamp}.md")
        
        return exports

