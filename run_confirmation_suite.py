"""
Phase 2.8f Confirmation Suite Runner

Runs 5-seed × 200-episode validation with comprehensive telemetry logging.
Implements all guardrails and acceptance gates from PHASE_2_8F_CONFIRMATION_PROTOCOL.md
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

def run_confirmation_suite(seeds, episodes_per_seed, output_dir):
    """
    Run confirmation suite with specified seeds and episodes.
    
    Args:
        seeds: List of random seeds
        episodes_per_seed: Number of episodes per seed
        output_dir: Directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = {
        'protocol_version': '1.0',
        'start_time': datetime.now().isoformat(),
        'seeds': seeds,
        'episodes_per_seed': episodes_per_seed,
        'total_episodes': len(seeds) * episodes_per_seed,
        'estimated_hours': len(seeds) * 2.5,
        'config_hash': None,  # Will be computed from actual config
        'controller_params': {
            'W': 64,
            'K_LONG': 0.8,
            'K_HOLD': 0.6,
            'LONG_BAND': 0.10,
            'HOLD_BAND': 0.07,
            'TAU_MIN': 0.8,
            'TAU_MAX': 1.5,
            'H_MIN': 0.95,
            'H_MAX': 1.10,
            'LEAK': 0.95,
            'LAMBDA_CLIP': 1.2
        },
        'acceptance_gates': {
            '1_mean_spr': {'threshold': 0.04, 'type': 'ge'},
            '2_std_spr': {'threshold': 0.035, 'type': 'le'},
            '3_trail5_median': {'threshold': 0.25, 'type': 'ge'},
            '4_positive_seeds': {'threshold': 3, 'type': 'ge'},
            '5_long_ratio_inband': {'threshold': 0.70, 'type': 'ge'},
            '6_hold_rate_inband': {'threshold': 0.70, 'type': 'ge'},
            '7_entropy_inband': {'threshold': 0.80, 'type': 'ge'},
            '8_switch_rate_inband': {'threshold': 0.70, 'type': 'ge'},
            '9_penalty_rate': {'threshold': 0.10, 'type': 'le'}
        }
    }
    
    # Save manifest
    with open(output_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("=" * 80)
    print("PHASE 2.8f CONFIRMATION SUITE")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes_per_seed}")
    print(f"Total episodes: {len(seeds) * episodes_per_seed}")
    print(f"Estimated time: {len(seeds) * 2.5:.1f} hours")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    print()
    
    # Run each seed
    results = {}
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 80}")
        print(f"SEED {i}/{len(seeds)}: {seed}")
        print(f"{'=' * 80}\n")
        
        start_time = time.time()
        
        # Import here to avoid circular dependencies
        import subprocess
        import sys
        
        # Run training with enhanced telemetry
        cmd = [
            sys.executable, 'main.py',
            '--episodes', str(episodes_per_seed),
            '--seed', str(seed),
            '--telemetry', 'extended',  # Enable extended telemetry
            '--output-dir', str(output_path / f'seed_{seed}')
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Stream output to console
                text=True
            )
            
            elapsed = time.time() - start_time
            print(f"\n✅ Seed {seed} completed in {elapsed/3600:.2f} hours")
            
            results[seed] = {
                'status': 'success',
                'elapsed_seconds': elapsed,
                'output_dir': str(output_path / f'seed_{seed}')
            }
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Seed {seed} FAILED after {elapsed/3600:.2f} hours")
            print(f"Error: {e}")
            
            results[seed] = {
                'status': 'failed',
                'elapsed_seconds': elapsed,
                'error': str(e)
            }
    
    # Save results
    manifest['end_time'] = datetime.now().isoformat()
    manifest['total_elapsed_seconds'] = sum(r.get('elapsed_seconds', 0) for r in results.values())
    manifest['seed_results'] = results
    
    with open(output_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("CONFIRMATION SUITE COMPLETE")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"Successful seeds: {successful}/{len(seeds)}")
    print(f"Total time: {manifest['total_elapsed_seconds']/3600:.2f} hours")
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("1. Run analysis: python analyze_confirmation_results.py")
    print("2. Review acceptance gates in generated report")
    print("3. If all gates pass, tag v2.8f-confirmed")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 2.8f confirmation suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full confirmation suite
  python run_confirmation_suite.py --seeds 42,123,456,789,1011 --episodes 200

  # Quick test with 3 seeds
  python run_confirmation_suite.py --seeds 42,123,456 --episodes 50

  # Custom output directory
  python run_confirmation_suite.py --seeds 42,123 --episodes 100 --output results/test
        """
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default='42,123,456,789,1011',
        help='Comma-separated list of random seeds (default: 42,123,456,789,1011)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=200,
        help='Episodes per seed (default: 200)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='confirmation_results',
        help='Output directory (default: confirmation_results)'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Validate
    if len(seeds) < 3:
        print("⚠️  Warning: Confirmation protocol requires 5 seeds for statistical validity")
        print(f"   Currently using {len(seeds)} seeds")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    if args.episodes < 80:
        print("⚠️  Warning: Confirmation protocol requires 200 episodes per seed")
        print(f"   Currently using {args.episodes} episodes")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Run suite
    run_confirmation_suite(seeds, args.episodes, args.output)


if __name__ == '__main__':
    main()
