"""
Phase 2.8f Confirmation Results Analyzer

Analyzes multi-seed confirmation run and checks all 9 acceptance gates.
Generates comprehensive report with recommendations.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class ConfirmationAnalyzer:
    """Analyzes confirmation suite results against acceptance gates."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.manifest = self._load_manifest()
        self.seed_data = self._load_seed_data()
        
    def _load_manifest(self) -> Dict:
        """Load confirmation suite manifest."""
        manifest_path = self.results_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def _load_seed_data(self) -> Dict[int, pd.DataFrame]:
        """Load episode metrics for each seed."""
        seed_data = {}
        
        for seed in self.manifest['seeds']:
            seed_dir = self.results_dir / f'seed_{seed}'
            metrics_file = seed_dir / 'episode_metrics.json'
            
            if not metrics_file.exists():
                print(f"⚠️  Warning: Metrics not found for seed {seed}")
                continue
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                seed_data[seed] = pd.DataFrame(data['episodes'])
        
        return seed_data
    
    def check_gate_1_mean_spr(self) -> Tuple[bool, Dict]:
        """Gate 1: Mean SPR ≥ +0.04"""
        mean_sprs = [df['SPR'].mean() for df in self.seed_data.values()]
        overall_mean = np.mean(mean_sprs)
        threshold = self.manifest['acceptance_gates']['1_mean_spr']['threshold']
        
        passed = overall_mean >= threshold
        
        return passed, {
            'gate': 'Mean SPR',
            'threshold': f'≥ {threshold}',
            'value': f'{overall_mean:.4f}',
            'per_seed': {seed: f'{mean:.4f}' for seed, mean in zip(self.seed_data.keys(), mean_sprs)},
            'passed': passed
        }
    
    def check_gate_2_std_spr(self) -> Tuple[bool, Dict]:
        """Gate 2: σ(mean SPR) ≤ 0.035"""
        mean_sprs = [df['SPR'].mean() for df in self.seed_data.values()]
        std_sprs = np.std(mean_sprs, ddof=1) if len(mean_sprs) > 1 else 0.0
        threshold = self.manifest['acceptance_gates']['2_std_spr']['threshold']
        
        passed = std_sprs <= threshold
        
        return passed, {
            'gate': 'Cross-Seed σ(SPR)',
            'threshold': f'≤ {threshold}',
            'value': f'{std_sprs:.4f}',
            'passed': passed
        }
    
    def check_gate_3_trail5_median(self) -> Tuple[bool, Dict]:
        """Gate 3: Trail-5 median ≥ +0.25"""
        trail5_medians = []
        for df in self.seed_data.values():
            if len(df) >= 5:
                trail5 = df['SPR'].iloc[-5:].median()
                trail5_medians.append(trail5)
        
        overall_median = np.median(trail5_medians) if trail5_medians else 0.0
        threshold = self.manifest['acceptance_gates']['3_trail5_median']['threshold']
        
        passed = overall_median >= threshold
        
        return passed, {
            'gate': 'Trail-5 Median SPR',
            'threshold': f'≥ {threshold}',
            'value': f'{overall_median:.4f}',
            'per_seed': {seed: f'{t5:.4f}' for seed, t5 in zip(self.seed_data.keys(), trail5_medians)},
            'passed': passed
        }
    
    def check_gate_4_positive_seeds(self) -> Tuple[bool, Dict]:
        """Gate 4: ≥3/5 seeds with positive mean SPR"""
        mean_sprs = [df['SPR'].mean() for df in self.seed_data.values()]
        positive_count = sum(1 for mean_spr in mean_sprs if mean_spr > 0)
        threshold = self.manifest['acceptance_gates']['4_positive_seeds']['threshold']
        
        passed = positive_count >= threshold
        
        return passed, {
            'gate': 'Positive Seeds',
            'threshold': f'≥ {threshold}/{len(self.seed_data)}',
            'value': f'{positive_count}/{len(self.seed_data)}',
            'passed': passed
        }
    
    def check_gate_5_long_ratio_inband(self) -> Tuple[bool, Dict]:
        """Gate 5: Long ratio [0.40, 0.60] in ≥70% of episodes"""
        inband_pcts = []
        
        for df in self.seed_data.values():
            if 'p_long_smoothed' in df.columns:
                inband = ((df['p_long_smoothed'] >= 0.40) & (df['p_long_smoothed'] <= 0.60)).mean()
                inband_pcts.append(inband)
        
        avg_inband = np.mean(inband_pcts) if inband_pcts else 0.0
        threshold = self.manifest['acceptance_gates']['5_long_ratio_inband']['threshold']
        
        passed = avg_inband >= threshold
        
        return passed, {
            'gate': 'Long Ratio In-Band',
            'threshold': f'≥ {threshold*100:.0f}%',
            'value': f'{avg_inband*100:.1f}%',
            'per_seed': {seed: f'{pct*100:.1f}%' for seed, pct in zip(self.seed_data.keys(), inband_pcts)},
            'passed': passed
        }
    
    def check_gate_6_hold_rate_inband(self) -> Tuple[bool, Dict]:
        """Gate 6: Hold rate [0.65, 0.79] in ≥70% of episodes"""
        inband_pcts = []
        
        for df in self.seed_data.values():
            if 'p_hold_smoothed' in df.columns:
                inband = ((df['p_hold_smoothed'] >= 0.65) & (df['p_hold_smoothed'] <= 0.79)).mean()
                inband_pcts.append(inband)
        
        avg_inband = np.mean(inband_pcts) if inband_pcts else 0.0
        threshold = self.manifest['acceptance_gates']['6_hold_rate_inband']['threshold']
        
        passed = avg_inband >= threshold
        
        return passed, {
            'gate': 'Hold Rate In-Band',
            'threshold': f'≥ {threshold*100:.0f}%',
            'value': f'{avg_inband*100:.1f}%',
            'per_seed': {seed: f'{pct*100:.1f}%' for seed, pct in zip(self.seed_data.keys(), inband_pcts)},
            'passed': passed
        }
    
    def check_gate_7_entropy_inband(self) -> Tuple[bool, Dict]:
        """Gate 7: Entropy [0.95, 1.10] in ≥80% of episodes"""
        inband_pcts = []
        
        for df in self.seed_data.values():
            if 'H_bits' in df.columns:
                inband = ((df['H_bits'] >= 0.95) & (df['H_bits'] <= 1.10)).mean()
                inband_pcts.append(inband)
        
        avg_inband = np.mean(inband_pcts) if inband_pcts else 0.0
        threshold = self.manifest['acceptance_gates']['7_entropy_inband']['threshold']
        
        passed = avg_inband >= threshold
        
        return passed, {
            'gate': 'Entropy In-Band',
            'threshold': f'≥ {threshold*100:.0f}%',
            'value': f'{avg_inband*100:.1f}%',
            'per_seed': {seed: f'{pct*100:.1f}%' for seed, pct in zip(self.seed_data.keys(), inband_pcts)},
            'passed': passed
        }
    
    def check_gate_8_switch_rate_inband(self) -> Tuple[bool, Dict]:
        """Gate 8: Switch rate [0.15, 0.19] in ≥70% of episodes"""
        inband_pcts = []
        
        for df in self.seed_data.values():
            if 'switch_rate' in df.columns:
                inband = ((df['switch_rate'] >= 0.15) & (df['switch_rate'] <= 0.19)).mean()
                inband_pcts.append(inband)
        
        avg_inband = np.mean(inband_pcts) if inband_pcts else 0.0
        threshold = self.manifest['acceptance_gates']['8_switch_rate_inband']['threshold']
        
        passed = avg_inband >= threshold
        
        return passed, {
            'gate': 'Switch Rate In-Band',
            'threshold': f'≥ {threshold*100:.0f}%',
            'value': f'{avg_inband*100:.1f}%',
            'per_seed': {seed: f'{pct*100:.1f}%' for seed, pct in zip(self.seed_data.keys(), inband_pcts)},
            'passed': passed
        }
    
    def check_gate_9_penalty_rate(self) -> Tuple[bool, Dict]:
        """Gate 9: Penalty/failsafe rate ≤ 10%"""
        penalty_rates = []
        
        for df in self.seed_data.values():
            if 'penalty_triggered' in df.columns:
                rate = df['penalty_triggered'].mean()
                penalty_rates.append(rate)
        
        avg_rate = np.mean(penalty_rates) if penalty_rates else 0.0
        threshold = self.manifest['acceptance_gates']['9_penalty_rate']['threshold']
        
        passed = avg_rate <= threshold
        
        return passed, {
            'gate': 'Penalty Rate',
            'threshold': f'≤ {threshold*100:.0f}%',
            'value': f'{avg_rate*100:.1f}%',
            'per_seed': {seed: f'{rate*100:.1f}%' for seed, rate in zip(self.seed_data.keys(), penalty_rates)},
            'passed': passed
        }
    
    def run_all_gates(self) -> Dict:
        """Run all acceptance gates and return results."""
        gate_results = {}
        
        # Run each gate
        gate_funcs = [
            self.check_gate_1_mean_spr,
            self.check_gate_2_std_spr,
            self.check_gate_3_trail5_median,
            self.check_gate_4_positive_seeds,
            self.check_gate_5_long_ratio_inband,
            self.check_gate_6_hold_rate_inband,
            self.check_gate_7_entropy_inband,
            self.check_gate_8_switch_rate_inband,
            self.check_gate_9_penalty_rate
        ]
        
        for i, gate_func in enumerate(gate_funcs, 1):
            passed, result = gate_func()
            gate_results[f'gate_{i}'] = result
        
        # Overall pass/fail
        all_passed = all(r['passed'] for r in gate_results.values())
        
        return {
            'gates': gate_results,
            'all_passed': all_passed,
            'passed_count': sum(1 for r in gate_results.values() if r['passed']),
            'total_count': len(gate_results)
        }
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive confirmation report."""
        results = self.run_all_gates()
        
        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PHASE 2.8f CONFIRMATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Results directory: {self.results_dir}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## SUMMARY")
        report_lines.append("")
        if results['all_passed']:
            report_lines.append("✅ **ALL GATES PASSED** - Ready for v2.8f-confirmed tag")
        else:
            report_lines.append(f"❌ **{results['total_count'] - results['passed_count']} GATE(S) FAILED**")
            report_lines.append("   Review tuning ladder in PHASE_2_8F_CONFIRMATION_PROTOCOL.md")
        
        report_lines.append("")
        report_lines.append(f"Gates passed: {results['passed_count']}/{results['total_count']}")
        report_lines.append("")
        
        # Gate details
        report_lines.append("## ACCEPTANCE GATES")
        report_lines.append("")
        
        for gate_id, gate_result in results['gates'].items():
            status = "✅ PASS" if gate_result['passed'] else "❌ FAIL"
            report_lines.append(f"### {gate_id.upper()}: {gate_result['gate']}")
            report_lines.append(f"**Status**: {status}")
            report_lines.append(f"**Threshold**: {gate_result['threshold']}")
            report_lines.append(f"**Measured**: {gate_result['value']}")
            
            if 'per_seed' in gate_result:
                report_lines.append("")
                report_lines.append("Per-seed breakdown:")
                for seed, value in gate_result['per_seed'].items():
                    report_lines.append(f"  - Seed {seed}: {value}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## RECOMMENDATIONS")
        report_lines.append("")
        
        if results['all_passed']:
            report_lines.append("### Next Steps:")
            report_lines.append("1. Tag release: `git tag -a v2.8f-confirmed -m \"Passed 5-seed confirmation\"`")
            report_lines.append("2. Push tag: `git push origin main --tags`")
            report_lines.append("3. Proceed to stress tests (see §6 in protocol)")
            report_lines.append("4. Begin paper trading preparation (see §8 in protocol)")
        else:
            report_lines.append("### Tuning Required:")
            report_lines.append("")
            
            # Specific recommendations based on failed gates
            for gate_id, gate_result in results['gates'].items():
                if not gate_result['passed']:
                    report_lines.append(f"**{gate_result['gate']}**:")
                    
                    if 'Long Ratio' in gate_result['gate']:
                        report_lines.append("  - Widen dead-zone: LONG_BAND = 0.12")
                        report_lines.append("  - Reduce gain: K_LONG = 0.6")
                        report_lines.append("  - Increase leak: LEAK = 0.997")
                    
                    elif 'Hold Rate' in gate_result['gate']:
                        report_lines.append("  - Increase gain: K_HOLD = 0.7")
                        report_lines.append("  - Add anti-stickiness after 40 HOLDs")
                    
                    elif 'Entropy' in gate_result['gate']:
                        report_lines.append("  - Faster correction: tau *= 1.07")
                        report_lines.append("  - Raise ceiling: TAU_MAX = 1.7")
                    
                    elif 'σ(SPR)' in gate_result['gate']:
                        report_lines.append("  - Smoother EWMA: W = 96")
                        report_lines.append("  - Reduce gains: K_LONG=0.6, K_HOLD=0.5")
                    
                    report_lines.append("")
            
            report_lines.append("After tuning, retest with 1 seed × 200 episodes before full suite.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Join report
        report = "\n".join(report_lines)
        
        # Print to console
        print(report)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(report)
            print(f"\nReport saved to: {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Phase 2.8f confirmation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='confirmation_results',
        help='Directory containing confirmation results (default: confirmation_results)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='confirmation_report.md',
        help='Output report file (default: confirmation_report.md)'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = ConfirmationAnalyzer(args.results_dir)
        analyzer.generate_report(args.output)
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
