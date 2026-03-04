#!/usr/bin/env python
"""
Simple test script for SequencerWidgetV2.

This script tests that the new sequencer can be instantiated and
can generate options sequences correctly.
"""

import sys
sys.path.insert(0, 'src')

from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.interactions.sequencer_v2 import SequencerWidgetV2

def test_sequencer_v2():
    """Test basic functionality of SequencerWidgetV2."""

    # Create base options
    base_options = ProjectionMatchingOptions()

    # Test 1: Create sequencer with basic options list
    print("Test 1: Creating SequencerWidgetV2...")
    basic_pma_settings = [
        "iterations",
        "high_pass_filter",
        "downsample",
        "downsample.scale",
    ]

    sequencer = SequencerWidgetV2(
        base_options,
        basic_options_list=basic_pma_settings,
    )
    print("✓ SequencerWidgetV2 created successfully")

    # Test 2: Generate options sequence (should have 1 block by default)
    print("\nTest 2: Generating options sequence...")
    options_sequence = sequencer.generate_options_sequence(base_options)
    print(f"✓ Generated sequence with {len(options_sequence)} option(s)")
    assert len(options_sequence) == 1, "Should have 1 option by default"

    # Test 3: Get changed settings sequence
    print("\nTest 3: Getting changed settings sequence...")
    changed_settings = sequencer.get_changed_settings_sequence()
    print(f"✓ Got changed settings: {changed_settings}")

    # Test 4: Load from list of dicts
    print("\nTest 4: Loading sequence from list of dicts...")
    test_settings = [
        {"high_pass_filter": 0.0123},
        {"downsample": {"scale": 16}},
        {"high_pass_filter": 0.0456, "iterations": 10},
    ]

    sequencer.generate_sequence_from_list_of_dicts(test_settings)
    options_sequence = sequencer.generate_options_sequence(base_options)
    print(f"✓ Generated sequence with {len(options_sequence)} blocks")
    assert len(options_sequence) == 3, f"Should have 3 blocks, got {len(options_sequence)}"

    # Verify the settings were applied
    print(f"  Block 1 - high_pass_filter: {options_sequence[0].high_pass_filter}")
    print(f"  Block 2 - downsample.scale: {options_sequence[1].downsample.scale}")
    print(f"  Block 3 - high_pass_filter: {options_sequence[2].high_pass_filter}, iterations: {options_sequence[2].iterations}")

    assert options_sequence[0].high_pass_filter == 0.0123, "Block 1 high_pass_filter mismatch"
    assert options_sequence[1].downsample.scale == 16, "Block 2 downsample.scale mismatch"
    assert options_sequence[2].high_pass_filter == 0.0456, "Block 3 high_pass_filter mismatch"
    assert options_sequence[2].iterations == 10, "Block 3 iterations mismatch"

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

if __name__ == "__main__":
    test_sequencer_v2()
