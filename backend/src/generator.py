"""
generator.py
------------
Synthetic Data Generator for Industrial Sensor Logs.

Purpose:
    Creates a dataset of "Instruction-Input-Output" triplets to fine-tune 
    Small Language Models (SLMs) for technical diagnostics.
    
    It simulates logs from Vaisala-style weather transmitters (e.g., WXT536, HMP155)
    and generates expert-level troubleshooting responses.

Output:
    JSONL file suitable for Hugging Face datasets.

Usage:
    python src/generator.py --count 1000
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Internal imports
try:
    from src.utils import get_logger, get_project_root
except ImportError:
    # Fallback for running script directly from src folder
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils import get_logger, get_project_root

logger = get_logger("DATAGEN")

# --- Domain Knowledge Base (Vaisala Context) ---

SENSORS = [
    "WXT536 Multi-Parameter Weather Sensor",
    "HMP155 Humidity and Temperature Probe",
    "DMT143 Dewpoint Transmitter",
    "PWD22 Present Weather Detector"
]

# Technical Error Scenarios
# Maps an error signature to a specific technical diagnosis
SCENARIOS = [
    {
        "type": "voltage_drop",
        "log_pattern": "V_IN: 8.4V [LOW] | REF_CHK: FAIL",
        "instruction": "Analyze the power subsystem logs for the weather station.",
        "response": "The input voltage (V_IN) has dropped to 8.4V, which is below the operating threshold of 10V-30V for the WXT536. This triggers a Reference Check (REF_CHK) failure. Recommendation: Check the solar charge controller or replace the backup battery immediately."
    },
    {
        "type": "heating_fail",
        "log_pattern": "HEAT_CURRENT: 0.00A | TEMP_AMB: -15.2C | STATUS: ICE_WARNING",
        "instruction": "Diagnose the heating system status based on current telemetry.",
        "response": "Critical failure in heating element. Ambient temperature is -15.2C, but Heater Current is 0.00A. The sensor is at risk of ice accumulation (ICE_WARNING), which will corrupt wind measurement data. Check the 24V heater power supply lines for continuity."
    },
    {
        "type": "calibration_drift",
        "log_pattern": "RH_RAW: 104.2% | OFFSET_CORR: MAX_LIMIT | FLAG: ERR_SATURATION",
        "instruction": "Review the humidity sensor calibration data.",
        "response": "The Relative Humidity (RH) sensor is reporting 104.2%, indicating saturation or chemical contamination. The internal offset correction algorithm has hit its maximum limit. The sensor likely requires chemical purging or the humidity chip needs replacement."
    },
    {
        "type": "protocol_error",
        "log_pattern": "NMEA 0183: $WIMWV,???,R,0.0,M,A*1B | CRC: MISMATCH",
        "instruction": "Interpret the communication error in the NMEA stream.",
        "response": "CRC Mismatch detected in NMEA 0183 sentence ($WIMWV). The wind angle field contains '???', suggesting the serial line (RS-485) has noise or a baud rate mismatch. Verify the termination resistor is installed on the data logger end."
    },
    {
        "type": "normal_op",
        "log_pattern": "SYS_STATUS: OK | V_IN: 12.1V | T: 22.4C | RH: 45% | UPTIME: 4500h",
        "instruction": "Check the system health summary.",
        "response": "System is operating normally. Input voltage (12.1V) is stable, and environmental readings are within expected ranges. Uptime of 4500 hours indicates a stable firmware state."
    }
]

def generate_timestamp() -> str:
    """Generates a random timestamp within the last 30 days."""
    delta = timedelta(days=random.randint(0, 30), seconds=random.randint(0, 86400))
    dt = datetime.now() - delta
    return dt.isoformat(timespec='seconds')

def create_sample(index: int) -> Dict[str, str]:
    """
    Constructs a single synthetic training example.
    
    Structure fits the 'Alpaca' or 'Phi-3' prompt format:
    - Instruction: What the user asks.
    - Input: The technical context (the log data).
    - Output: The expert answer.
    """
    scenario = random.choice(SCENARIOS)
    sensor = random.choice(SENSORS)
    
    # Add random variations to make the model robust to noise
    timestamp = generate_timestamp()
    log_id = f"LOG-{10000 + index}"
    
    # Construct the raw log input
    # We combine the ID, Timestamp, Sensor Name, and the technical pattern
    log_input = f"[{timestamp}] ID:{log_id} DEVICE:{sensor} >>> {scenario['log_pattern']}"
    
    return {
        "instruction": scenario['instruction'],
        "input": log_input,
        "output": scenario['response']
    }

def main():
    """
    Main execution entry point.
    """
    parser = argparse.ArgumentParser(description="NanoSentri Synthetic Data Generator")
    parser.add_argument("--count", type=int, default=500, help="Number of samples to generate")
    args = parser.parse_args()

    # Setup paths
    root_dir = get_project_root()
    output_path = root_dir / "data" / "processed" / "vaisala_synthetic_train.jsonl"
    
    logger.info(f"Starting data generation for {args.count} samples...")
    
    dataset = []
    
    # Generation Loop
    for i in range(args.count):
        sample = create_sample(i)
        dataset.append(sample)
    
    # Write to JSONL (JSON Lines) - Preferred format for Streaming datasets
    logger.info(f"Writing data to {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                json.dump(entry, f)
                f.write('\n')
        
        logger.info("Generation complete. Success.")
        logger.info(f"Preview of first sample:\n{json.dumps(dataset[0], indent=2)}")
        
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()