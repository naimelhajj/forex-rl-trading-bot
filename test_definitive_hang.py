"""
DEFINITIVE HANG TEST - Print every single operation.
If this hangs, we'll know EXACTLY where.
"""
import sys
import time

def p(msg):
    print(msg, flush=True)

p("START")
p("1")
from config import Config
p("2")
config = Config()
p("3")
from data_loader import DataLoader  
p("4")
loader = DataLoader(config.data_dir, "EURUSD")
p("5 - calling load_split_data")
train_df, val_df, _ = loader.load_split_data(0.7, 0.15)
p(f"6 - data loaded, train={len(train_df)} bars")

p("7 - importing FeatureEngineer")
from features import FeatureEngineer
p("8 - creating FeatureEngineer()")
eng = FeatureEngineer()
p("9 - calling compute_all_features on TRAIN")

t0 = time.time()
train_df = eng.compute_all_features(train_df)
p(f"10 - TRAIN features done in {time.time()-t0:.3f}s")

p("11 - calling compute_all_features on VAL")  
t0 = time.time()
val_df = eng.compute_all_features(val_df)
p(f"12 - VAL features done in {time.time()-t0:.3f}s")

p("DONE - No hang detected")
