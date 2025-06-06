# upload_dataset.py (diperbaiki lagi)
import modal
import os
from pathlib import Path

nfs_name = "lipreading-dataset-nfs"
local_dataset_dir = "idev1_roi_80_116_175_211_npy_gray_pkl_jpeg"
remote_dataset_dir = f"/data/{local_dataset_dir}"

app = modal.App("dataset-uploader")
nfs = modal.NetworkFileSystem.from_name(nfs_name, create_if_missing=True)

@app.function(
    # --- MODIFIKASI: Menggunakan path absolut "/data" ---
    network_file_systems={"/data": nfs},
    timeout=1800
)
def upload():
    # Pastikan direktori lokal ada sebelum mencoba mengunggah
    if not os.path.isdir(local_dataset_dir):
        print(f"Error: Direktori lokal '{local_dataset_dir}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan 'python scripts/prepare_idev1.py' terlebih dahulu.")
        return

    print(f"Memeriksa apakah direktori tujuan '{remote_dataset_dir}' sudah ada...")
    if not os.path.exists(remote_dataset_dir):
        print("Direktori belum ada, memulai proses unggah...")
        import shutil
        shutil.copytree(local_dataset_dir, remote_dataset_dir)
        print(f"Berhasil mengunggah dataset ke {remote_dataset_dir}")
    else:
        print("Dataset sudah ada di NFS, proses unggah dilewati.")