#!/usr/bin/env python3
"""Controls Analyzer - LAMDA v2.6+

Summarizes MIDI control changes:
- Pitch bend range (min/max values)
- RPN (Registered Parameter Number) detection
- CC (Control Change) usage summary per controller

Design: NO-OP safe - returns default values for empty MIDI.
"""

from __future__ import annotations
from typing import Any, Dict

# RPN-related CC numbers
RPN_CC_NUMS = {100, 101, 6, 38}


def analyze_controls(pm: Any) -> Dict[str, Any]:
    """Analyze MIDI control changes (PB/RPN/CC).

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI data object.

    Returns
    -------
    Dict[str, Any]
        {
            "pb_range": [int, int],  # [min, max] pitch bend values
            "cc_summary": {
                "cc_num": {"min": int, "max": int}, ...
            },
            "rpn_seen": bool  # Whether RPN-related CCs detected
        }

    Examples
    --------
    >>> controls = analyze_controls(midi_data)
    >>> controls["pb_range"]
    [-2048, 2048]
    >>> controls["rpn_seen"]
    True
    """
    pb_min, pb_max = 0, 0
    rpn_seen = False
    cc_summary: Dict[str, Dict[str, int]] = {}

    for ins in pm.instruments:
        # Pitch bend analysis
        for pb in ins.pitch_bends:
            pb_min = min(pb_min, int(pb.pitch))
            pb_max = max(pb_max, int(pb.pitch))

        # Control change analysis
        for cc in ins.control_changes:
            k = str(cc.number)
            stat = cc_summary.setdefault(k, {"min": 127, "max": 0})
            v = int(cc.value)
            stat["min"] = min(stat["min"], v)
            stat["max"] = max(stat["max"], v)

            # RPN detection (CC 100, 101, 6, 38)
            if cc.number in RPN_CC_NUMS:
                rpn_seen = True

    return {
        "pb_range": [int(pb_min), int(pb_max)],
        "cc_summary": cc_summary,
        "rpn_seen": bool(rpn_seen),
        "integrity": 1.0,
    }
