# demo/run_all.py - Unleash the King on all cursed images
from ocr_chain import OCRChain
from pathlib import Path

chain = OCRChain()
demo_dir = Path("demo")
out_file = demo_dir / "RESULTS.md"

print("FEEDING THE KING 5 IMPOSSIBLE SCANS...")

with open(out_file, "w", encoding="utf-8") as f:
    f.write("# ARCHEOGODZILLA V2.0 – THE KING VS. THE IMPOSSIBLE\n\n")
    f.write("| Image | GODZILLA SAYS | Confidence | Engine |\n")
    f.write("|-------|---------------|------------|--------|\n")

    for img_path in sorted(demo_dir.glob("*.jpg")) + sorted(demo_dir.glob("*.png")) + sorted(demo_dir.glob("*.tif")):
        if img_path.name.startswith("RESULT") or img_path.name == ".gitkeep":
            continue
        print(f"  → Devouring {img_path.name}")
        img = chain.preprocess(str(img_path))
        text, results = chain.extract_text(img)
        best = max(results, key=lambda k: results[k]['confidence'])
        conf = results[best]['confidence']
        engine = best
        safe_text = text.replace("|", "│").replace("\n", " ").strip()
        if len(safe_text) > 200:
            safe_text = safe_text[:197] + "..."
        f.write(f"| {img_path.name} | {safe_text} | {conf:.3f} | {engine} |\n")

print("DONE. RESULTS.md created in demo/ – push it and watch the nerds kneel.")
