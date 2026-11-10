"""Regenerate reports without running inference; same callable used by notebook."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.reporting import render_records

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    print(render_records(a.predictions, a.output_dir)['argmax'])

if __name__ == '__main__':
    main()
