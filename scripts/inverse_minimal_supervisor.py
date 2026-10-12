"""Confirmed single InverseRenderer run using the established generic supervisor."""
import argparse
from pathlib import Path
import sys

from forward_minimal_supervisor import Policy, supervise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.confirmed:
        parser.error('Explicit user confirmation is required before GPU/model loading.')
    worker=Path(__file__).with_name('inverse_minimal_once.py')
    result=supervise([sys.executable,'-B','-u',str(worker),'--confirmed',
                      '--output-dir',str(args.output_dir.resolve())],args.output_dir,Policy())
    print(__import__('json').dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
