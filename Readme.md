# DigiFlash-Archiver 

Quiet, boring, reliable digitization tools for turning paper mountains - and microfiche trays - into searchable archives. '80s vibes, modern under the hood. No drama. Just done. 

## What It Does 

- Scans paper, slides, microfiche → clean text
- Tags with timestamps
- Outputs searchable Markdown
- Auto-backs up to Git/IPFS (optional)
No cloud.
No AI hype.
Just Dene grit. 

## Quick Start bash 

# 1. Clone git clone https://github.com/lyleantoine-collab/DigiFlash-Archiver.git cd DigiFlash-Archiver 
# 2. Install pip install opencv-python pytesseract pillow scikit-image 
# 3. Paper mode python scripts/digi_flash.py --folder scans/ --mode paper 
# 4. Microfiche mode python scripts/digi_flash.py --folder fiche_tray/ --mode film 

## Microfiche Mode 

Drop frames in a folder. 
Run --mode film. 
Deblurs dots, OCRs text, timestamps each page. 
Output: .md per frame. 
Backup: python scripts/batch_backup.py. 
Handles faded ink, '50s records, legacy dust - no names. 
No noise. 
Just truth. 

## Roadmap (Ongoing) 
- [ ] Add Whisper for voice-dirty scans
- [ ] IPFS auto-pin
- [ ] Mobile mode – run from phone, scan on-site
- Microfiche support (live)

Fork. 
Tweak. 
Archive. 

Nakehk'o Innovations | Goulds, NL | #ArchiveSovereignty
