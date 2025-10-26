
#!/usr/bin/env python3
# stage1_config_validator.py
import sys, yaml, re

def validate(cfg):
    issues = []
    def expect_keys(d, keys, where):
        for k in keys:
            if k not in d:
                issues.append(f"[YAML] Missing key '{k}' in {where}")
    expect_keys(cfg, ["version","roots","policy","ranges","id_rules","ok_meta","logging"], "root")
    if cfg.get("version") != 2:
        issues.append(f"[YAML] version must be 2 (got {cfg.get('version')})")
    roots = cfg.get("roots", {})
    expect_keys(roots, ["base","midi_in","midi_out","exclude_dirs"], "roots")
    policy = cfg.get("policy", {})
    expect_keys(policy, ["tempo_bpm_clip","tempo_min_hold_beats","timesig_rescue","drum_normalize","bar_split_long_notes"], "policy")
    ranges = cfg.get("ranges", {})
    expect_keys(ranges, ["pitch","vel","dur_ticks"], "ranges")
    for k in ["pitch","vel","dur_ticks"]:
        v = ranges.get(k, [])
        if not (isinstance(v, list) and len(v) == 2 and v[0] < v[1]):
            issues.append(f"[YAML] ranges.{k} must be [min,max] with min<max")
    if "${base}" in open(sys.argv[1], "r", encoding="utf-8").read():
        print("[INFO] Placeholder ${base} detected. Ensure your loader expands it safely.")
    return issues

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "stage1_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    issues = validate(cfg)
    if issues:
        print("\n".join(issues))
        sys.exit(1)
    print("OK: stage1_config.yaml passed basic validation.")
