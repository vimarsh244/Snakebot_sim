import fitz
import os

out_dir = r"C:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images"

papers = {
    "thesis": r"c:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images\2407.10300v1_very-good_and_important_thesis_doc.pdf",
    "ifac": r"c:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images\1-s2.0-S2405896320333772-main.pdf",
    "naish": r"c:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images\naish24a.pdf",
    "serpentine": r"c:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images\Reinforcement_Learning_of_Serpentine_Locomotion_for_a_Snake_Robot.pdf",
    "sensors": r"c:\Users\Vimarsh\Desktop\ERC\Snakebot\reference_images\sensors-22-09867.pdf",
}

for name, path in papers.items():
    try:
        doc = fitz.open(path)
        out_path = os.path.join(out_dir, f"{name}_text.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== {os.path.basename(path)} ===\n")
            f.write(f"Pages: {len(doc)}\n\n")
            for i in range(len(doc)):
                text = doc[i].get_text()
                f.write(f"--- Page {i+1} ---\n")
                f.write(text + "\n")
            doc.close()
        print(f"OK: {name} -> {out_path}")
    except Exception as e:
        print(f"Error: {name}: {e}")
