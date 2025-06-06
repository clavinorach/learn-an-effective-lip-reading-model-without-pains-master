import modal
import os
import subprocess

# --- Konfigurasi ---
NFS_NAME = "lipreading-dataset-nfs"
APP_NAME = "lipreading-training-app"

# --- Definisi App Modal ---
app = modal.App(APP_NAME)

# --- Definisi Environment Container ---
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04")
    .apt_install("libgl1", "libturbojpeg0")
    .pip_install(
        "torch==2.3.0",
        "numpy",
        "opencv-python-headless",
        "PyTurboJPEG",
        "einops",
    )
)

# --- Definisi Network File System ---
nfs = modal.NetworkFileSystem.from_name(NFS_NAME)

# --- Fungsi Training Utama ---
@app.function(
    image=cuda_image,
    gpu="H200",
    network_file_systems={"/data": nfs},
    mounts=[
        # --- PENDEKATAN BARU: Mount seluruh direktori proyek saat ini ---
        # Ini akan menyalin semua file (termasuk .py, .txt, dll.) ke /root/ di container.
        modal.Mount.from_local_dir(".", remote_path="/root")
    ],
    timeout=43200
)
def train():
    """
    Fungsi ini dijalankan di dalam container Modal untuk melakukan training.
    """
    # Mengatur direktori kerja di dalam container agar sama dengan direktori proyek.
    # Ini memastikan semua impor relatif seperti 'from utils import ...' berfungsi.
    os.chdir("/root")

    print("--- Memulai Proses Training di Modal ---")
    print(f"Direktori kerja saat ini di dalam container: {os.getcwd()}")

    # --- 1. Modifikasi Path Dataset ---
    base_dataset_dir = "idev1_roi_80_116_175_211_npy_gray_pkl_jpeg"
    remote_dataset_path = f"/data/{base_dataset_dir}"
    dataset_script_path = "utils/dataset.py" # Path relatif sekarang berfungsi

    print(f"Memodifikasi {dataset_script_path} untuk menggunakan path NFS: {remote_dataset_path}")
    try:
        with open(dataset_script_path, "r") as f:
            content = f.read()
        
        new_content = content.replace(f"os.path.join('idev1_roi_80_116_175_211_npy_gray_pkl_jpeg'", f"os.path.join('{remote_dataset_path}'")
        
        with open(dataset_script_path, "w") as f:
            f.write(new_content)
        print("Path dataset berhasil dimodifikasi.")
    except FileNotFoundError:
        print(f"ERROR: Tidak dapat menemukan {dataset_script_path}. Pastikan file tersebut ada di direktori Anda.")
        return # Hentikan eksekusi jika file tidak ditemukan

    # --- 2. Persiapan Direktori Checkpoint ---
    checkpoint_dir = "/data/checkpoints/lipreading-model/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Hasil checkpoints akan disimpan di: {checkpoint_dir}")

    # --- 3. Menyusun dan Menjalankan Perintah Training ---
    cmd = [
        "python", "-u", "main_visual.py",
        "--lr=3e-4", "--batch_size=128", "--num_workers=16",
        "--max_epoch=120", "--test=False", f"--save_prefix={checkpoint_dir}",
        "--n_class=8", "--dataset=idev1", "--border=True",
        "--mixup=True", "--label_smooth=True", "--se=True"
    ]
    
    print("\nPerintah yang akan dijalankan:")
    print(" ".join(cmd))
    print("\n--- Log Training ---")

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

    if process.returncode == 0:
        print("\n--- Training Selesai dengan Sukses ---")
    else:
        print(f"\n--- Training Gagal dengan Kode Error: {process.returncode} ---")

@app.local_entrypoint()
def main():
    train.remote()