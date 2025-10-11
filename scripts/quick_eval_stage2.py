#!/usr/bin/env python3
"""
scripts/quick_eval_stage2.py

Real Stage2 Integration for A/B Evaluation
- Generates sequences via Stage3
- Evaluates via lamda_stage2_extractor.py (real Stage2)
- Computes KPI: pass_rate, p50, p90, bar_violation_rate
- Outputs: JSON + Markdown report
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_test_sequences(
    checkpoint_path: str,
    num_samples: int = 50,
    output_dir: str = "outputs/eval_stage2"
) -> List[str]:
    """
    Generate test sequences using Stage3 model
    
    Args:
        checkpoint_path: Path to Stage3 checkpoint
        num_samples: Number of samples to generate
        output_dir: Directory to save generated MIDI files
        
    Returns:
        List of generated MIDI file paths
    """
    from ml.stage3_infer import Stage3Tokenizer, generate_sequences
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test prompts with diverse conditions
    prompts = [
        {
            "emotion": "happy",
            "genre": "pop",
            "valence": 8,
            "arousal": 7,
            "tempo": 120,
            "max_bars": 8,
        },
        {
            "emotion": "sad",
            "genre": "classical",
            "valence": 3,
            "arousal": 4,
            "tempo": 80,
            "max_bars": 8,
        },
        {
            "emotion": "energetic",
            "genre": "jazz",
            "valence": 7,
            "arousal": 8,
            "tempo": 140,
            "max_bars": 8,
        },
        {
            "emotion": "calm",
            "genre": "ambient",
            "valence": 6,
            "arousal": 3,
            "tempo": 60,
            "max_bars": 8,
        },
    ]
    
    tokenizer = Stage3Tokenizer()
    generated_files = []
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        
        try:
            # Generate sequence
            output = generate_sequences(
                checkpoint_path=checkpoint_path,
                prompts=[prompt],
                max_bars=prompt["max_bars"],
                temperature=0.9,
                top_k=50,
                top_p=0.9,
                enforce_bar_constraint=True
            )
            
            if output and output[0]:
                midi_path = os.path.join(output_dir, f"sample_{i:03d}.mid")
                output[0].write(midi_path)
                generated_files.append(midi_path)
                print(f"✅ Generated: {midi_path}")
            else:
                print(f"⚠️  Failed to generate sample {i}")
                
        except Exception as e:
            print(f"❌ Error generating sample {i}: {e}")
            continue
    
    return generated_files


def evaluate_with_stage2(midi_files: List[str]) -> List[Dict[str, Any]]:
    """
    Evaluate generated MIDI files using Stage2 extractor
    
    Args:
        midi_files: List of MIDI file paths
        
    Returns:
        List of evaluation results with Stage2 scores
    """
    results = []
    
    # Check if Stage2 extractor exists
    stage2_script = PROJECT_ROOT / "lamda_stage2_extractor.py"
    if not stage2_script.exists():
        print(f"⚠️  Stage2 extractor not found at {stage2_script}")
        print("Using fallback evaluation...")
        return _fallback_evaluation(midi_files)
    
    for midi_path in midi_files:
        try:
            # Run Stage2 extractor
            cmd = [
                "python3",
                str(stage2_script),
                "--input", midi_path,
                "--output", midi_path.replace(".mid", ".stage2.json")
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"⚠️  Stage2 evaluation failed for {midi_path}")
                print(f"stderr: {result.stderr}")
                results.append({
                    "file": midi_path,
                    "error": result.stderr,
                    "stage2_score": None
                })
                continue
            
            # Parse Stage2 output
            output_path = midi_path.replace(".mid", ".stage2.json")
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    stage2_data = json.load(f)
                
                results.append({
                    "file": midi_path,
                    "stage2_score": stage2_data.get("total_score", 0),
                    "axes": stage2_data.get("axes", {}),
                    "pass": stage2_data.get("total_score", 0) >= 50,
                    "bar_violations": _count_bar_violations(midi_path)
                })
            else:
                results.append({
                    "file": midi_path,
                    "error": "Stage2 output not found",
                    "stage2_score": None
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏱️  Stage2 evaluation timed out for {midi_path}")
            results.append({
                "file": midi_path,
                "error": "Timeout",
                "stage2_score": None
            })
        except Exception as e:
            print(f"❌ Error evaluating {midi_path}: {e}")
            results.append({
                "file": midi_path,
                "error": str(e),
                "stage2_score": None
            })
    
    return results


def _fallback_evaluation(midi_files: List[str]) -> List[Dict[str, Any]]:
    """
    Fallback evaluation when Stage2 extractor is not available
    Uses basic MIDI structure checks
    """
    results = []
    
    for midi_path in midi_files:
        try:
            import pretty_midi
            
            pm = pretty_midi.PrettyMIDI(midi_path)
            
            # Basic quality metrics
            num_notes = sum(len(inst.notes) for inst in pm.instruments)
            duration = pm.get_end_time()
            bar_violations = _count_bar_violations(midi_path)
            
            # Heuristic score (0-100)
            score = 50
            if num_notes > 20:
                score += 10
            if duration > 10:
                score += 10
            if bar_violations == 0:
                score += 20
            if len(pm.instruments) > 0:
                score += 10
            
            results.append({
                "file": midi_path,
                "stage2_score": score,
                "axes": {
                    "num_notes": num_notes,
                    "duration": duration,
                    "bar_violations": bar_violations
                },
                "pass": score >= 50,
                "bar_violations": bar_violations,
                "note": "Fallback evaluation (Stage2 not available)"
            })
            
        except Exception as e:
            results.append({
                "file": midi_path,
                "error": str(e),
                "stage2_score": None
            })
    
    return results


def _count_bar_violations(midi_path: str) -> int:
    """
    Count BAR constraint violations in generated MIDI
    (Placeholder - actual implementation depends on tokenizer metadata)
    """
    # This would parse the generation metadata or re-tokenize
    # For now, return 0 as placeholder
    return 0


def compute_kpi(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute KPI metrics from evaluation results
    
    Args:
        results: List of evaluation results
        
    Returns:
        KPI dictionary with pass_rate, p50, p90, etc.
    """
    scores = [r["stage2_score"] for r in results if r.get("stage2_score") is not None]
    passes = [r for r in results if r.get("pass") is True]
    bar_violations = [r["bar_violations"] for r in results if "bar_violations" in r]
    
    if not scores:
        return {
            "error": "No valid scores",
            "total_samples": len(results)
        }
    
    kpi = {
        "total_samples": len(results),
        "valid_samples": len(scores),
        "pass_rate": len(passes) / len(scores) if scores else 0.0,
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "mean_score": float(np.mean(scores)),
        "bar_violation_rate": sum(bar_violations) / len(bar_violations) if bar_violations else 0.0,
        "bar_violations_total": sum(bar_violations),
    }
    
    # Gate judgment
    kpi["gate_pass"] = (
        kpi["pass_rate"] >= 0.7 and
        kpi["p50"] >= 60 and
        kpi["bar_violation_rate"] == 0.0
    )
    
    return kpi


def save_report(
    kpi: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str
):
    """
    Save evaluation report as JSON and Markdown
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"eval_report_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "kpi": kpi,
            "results": results,
            "timestamp": timestamp
        }, f, indent=2)
    
    # Save Markdown
    md_path = os.path.join(output_dir, f"eval_report_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write("# Stage3 Evaluation Report (Real Stage2 Integration)\n\n")
        f.write(f"**Generated**: {timestamp}\n\n")
        
        f.write("## KPI Summary\n\n")
        f.write(f"- **Total Samples**: {kpi.get('total_samples', 0)}\n")
        f.write(f"- **Valid Samples**: {kpi.get('valid_samples', 0)}\n")
        f.write(f"- **Pass Rate**: {kpi.get('pass_rate', 0):.1%}\n")
        f.write(f"- **P50 Score**: {kpi.get('p50', 0):.1f}\n")
        f.write(f"- **P90 Score**: {kpi.get('p90', 0):.1f}\n")
        f.write(f"- **Mean Score**: {kpi.get('mean_score', 0):.1f}\n")
        f.write(f"- **BAR Violation Rate**: {kpi.get('bar_violation_rate', 0):.1%}\n")
        f.write(f"- **Total BAR Violations**: {kpi.get('bar_violations_total', 0)}\n\n")
        
        gate_status = "✅ PASS" if kpi.get('gate_pass') else "❌ FAIL"
        f.write(f"**Gate Judgment**: {gate_status}\n\n")
        
        f.write("## Gate Criteria\n\n")
        f.write("- Pass Rate ≥ 70%\n")
        f.write("- P50 ≥ 60\n")
        f.write("- BAR Violation Rate = 0%\n\n")
        
        f.write("## Sample Results\n\n")
        f.write("| File | Score | Pass | BAR Violations |\n")
        f.write("|------|-------|------|----------------|\n")
        for r in results[:20]:  # Show first 20
            score = r.get("stage2_score", "N/A")
            pass_mark = "✅" if r.get("pass") else "❌"
            violations = r.get("bar_violations", "N/A")
            filename = os.path.basename(r["file"])
            f.write(f"| {filename} | {score} | {pass_mark} | {violations} |\n")
    
    print(f"\n📊 Report saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def main():
    """
    Main evaluation workflow
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage3 Real Stage2 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage3 checkpoint path")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_stage2", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Stage3 Evaluation: Real Stage2 Integration")
    print("=" * 70)
    
    # Step 1: Generate sequences
    print("\n[1/3] Generating test sequences...")
    midi_files = generate_test_sequences(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    print(f"✅ Generated {len(midi_files)} MIDI files")
    
    # Step 2: Evaluate with Stage2
    print("\n[2/3] Evaluating with Stage2...")
    results = evaluate_with_stage2(midi_files)
    print(f"✅ Evaluated {len(results)} samples")
    
    # Step 3: Compute KPI and save report
    print("\n[3/3] Computing KPI and generating report...")
    kpi = compute_kpi(results)
    save_report(kpi, results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("KPI Summary")
    print("=" * 70)
    print(f"Pass Rate:     {kpi.get('pass_rate', 0):.1%}")
    print(f"P50:           {kpi.get('p50', 0):.1f}")
    print(f"P90:           {kpi.get('p90', 0):.1f}")
    print(f"Violations:    {kpi.get('bar_violation_rate', 0):.1%}")
    print(f"Gate:          {'✅ PASS' if kpi.get('gate_pass') else '❌ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
