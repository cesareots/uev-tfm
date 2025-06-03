from pathlib import Path

resume_checkpoint_file = "20250601-232325/model_CNN3D_best.pth"
if Path(resume_checkpoint_file).is_absolute():
    print("yes 1")

resume_checkpoint_file = "d:/uev-tfm/models/CNN3D/20250601-232325/model_CNN3D_best.pth"
if Path(resume_checkpoint_file).is_absolute():
    print("yes 2")
