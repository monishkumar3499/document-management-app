"""
Interactive test for the trained RoBERTa model stored in bert_model/saved_model.

Usage:
    python test.py
Paste or type a paragraph (multiple lines allowed)
Press ENTER on an empty line to classify
Type 'exit' or 'quit' to stop
"""

from classifier import DepartmentClassifier


def read_paragraph():
    """
    Reads multi-line paragraph input from user.
    Ends when user enters an empty line.
    """
    print("\nEnter paragraph (press ENTER twice to submit):")
    lines = []
    while True:
        line = input()
        if line.strip().lower() in {"exit", "quit"}:
            return "EXIT"
        if line.strip() == "":
            break
        lines.append(line)
    return " ".join(lines).strip()


def main():
    print("\nInitializing Department Classifier...")
    clf = DepartmentClassifier()
    print("✅ Model loaded successfully\n")

    print("=== Department Classification (Paragraph Mode) ===")
    print("• Paste/type multiple lines")
    print("• Press ENTER on empty line to classify")
    print("• Type 'exit' or 'quit' to stop\n")

    while True:
        text = read_paragraph()

        if text == "EXIT":
            print("\nExiting classifier. Goodbye 👋")
            break

        if not text:
            print("⚠️ Please enter some text.\n")
            continue

        # Run classification
        result = clf.classify_text(text)

        print("\n📌 Prediction Result")
        print("Final Department      :", result["final_department"])
        print("Top Model Predictions :", result["top_model_preds"])
        # no keyword scores anymore – pure model output
        print("-" * 80)


if __name__ == "__main__":
    main()
