"""Summarize micro-averaged MMLU accuracy and optional dense-relative deltas."""
try:
    from .summarize_accuracy import main
except ImportError:
    from summarize_accuracy import main

if __name__ == "__main__":
    main(["mmlu"])
