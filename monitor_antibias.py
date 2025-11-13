"""
Monitor LONG bias fix progress during training
"""
import json
import time
from pathlib import Path

results_dir = Path('quick_test_antibias')
metrics_file = results_dir / 'episode_metrics.json'

print("Monitoring LONG bias fix...")
print("=" * 60)

last_episode_count = 0
while True:
    if metrics_file.exists():
        try:
            with open(metrics_file, encoding='utf-8') as f:
                data = json.load(f)
            
            episodes = data['episodes']
            if len(episodes) > last_episode_count:
                # Show latest episodes
                for ep in episodes[last_episode_count:]:
                    ep_num = ep['episode']
                    p_long = ep['p_long_smoothed']
                    lambda_long = ep['lambda_long']
                    long_entries = ep.get('long_entries', 0)
                    
                    status = "✅" if 0.30 <= p_long <= 0.70 else "❌"
                    floor_active = "🔧" if ep_num <= 60 else ""
                    
                    print(f"{status} Ep {ep_num:3d}: p_long={p_long:.3f}, "
                          f"λ_long={lambda_long:+.2f}, LONG entries={long_entries} {floor_active}")
                
                last_episode_count = len(episodes)
                
                # Summary every 5 episodes
                if len(episodes) >= 5 and len(episodes) % 5 == 0:
                    recent = episodes[-5:]
                    avg_p_long = sum(e['p_long_smoothed'] for e in recent) / 5
                    avg_entries = sum(e.get('long_entries', 0) for e in recent) / 5
                    print(f"\n📊 Last 5 eps avg: p_long={avg_p_long:.3f}, "
                          f"LONG entries/ep={avg_entries:.1f}\n")
        
        except (json.JSONDecodeError, KeyError):
            pass  # File being written, try again
    
    time.sleep(2)  # Check every 2 seconds
    
    # Exit after 10 episodes
    if last_episode_count >= 10:
        print("\n" + "=" * 60)
        print("✅ Test complete!")
        
        # Final summary
        with open(metrics_file, encoding='utf-8') as f:
            data = json.load(f)
        episodes = data['episodes']
        
        p_longs = [ep['p_long_smoothed'] for ep in episodes]
        long_entries = [ep.get('long_entries', 0) for ep in episodes]
        
        print(f"\nFinal stats ({len(episodes)} episodes):")
        print(f"  p_long: min={min(p_longs):.3f}, max={max(p_longs):.3f}, "
              f"mean={sum(p_longs)/len(p_longs):.3f}")
        print(f"  LONG entries: min={min(long_entries)}, max={max(long_entries)}, "
              f"mean={sum(long_entries)/len(long_entries):.1f}")
        print(f"  Episodes with p_long>0.10: {sum(1 for p in p_longs if p > 0.10)}/{len(p_longs)}")
        
        break
